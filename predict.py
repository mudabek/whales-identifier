import argparse
import yaml
import gc
import pathlib
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import hnswlib

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import HappyWhaleDataset
import models
from utils import prepare_loaders



class Predictor:
    def __init__(self, path_to_config):
        with open(path_to_config) as f:
            config = yaml.safe_load(f)

        # Hyperparameters
        self.config = config
        self.device = torch.device(config['device'])
        self.num_neighb = config['num_neighbors']
        self.title_run = config['title_run']
        self.test_batch_size = config['test_batch_size']
        self.num_workers = config['num_workers']
        self.mode = config['mode']
        self.embedding_size = config['embedding_size']

        # Check device
        print(f'Predicting on {self.device}')

        

    def load_dataloaders(self):
        # Data transforms:
        test_transforms = A.Compose([
            A.Resize(self.config['img_size'], self.config['img_size']),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ),
            ToTensorV2()], p=1.)
        data_transforms = {}
        data_transforms['train'] = test_transforms
        data_transforms['valid'] = test_transforms

        # Train and test datasets
        loaders = prepare_loaders(data_transforms, self.config)
        self.total_dataset_size = loaders['total_size'] 
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.test_loader = loaders['test']


    def load_eval_model(self):
        # Model
        self.model = models.TorchModel(self.config)
        self.model.load_state_dict(torch.load(self.config['pred_model_path'], map_location=self.device))
        self.model.to(self.device)
        self.model.eval()


    def load_encoder(self):
        # Load label decoder
        with open(f'label_encoder_{self.mode}.pkl', 'rb') as f:
            self.encoder = joblib.load(f)

    
    @torch.inference_mode()
    def get_embeddings(self, dataloader, stage):
        all_image_names = []
        all_embeddings = []
        all_targets = []

        for batch in tqdm(dataloader, desc=f"Creating {stage} embeddings"):
            image_names = batch['image_id']
            images = batch['image'].to(self.device)
            targets = batch['label'].to(self.device)

            embeddings = self.model.extract(images)

            all_image_names.append(image_names)
            all_embeddings.append(embeddings.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        all_image_names = np.concatenate(all_image_names)
        all_embeddings = np.vstack(all_embeddings)
        all_targets = np.concatenate(all_targets)

        all_embeddings = normalize(all_embeddings, axis=1, norm="l2")
        all_targets = self.encoder.inverse_transform(all_targets)

        return all_image_names, all_embeddings, all_targets


    def create_and_search_index_hnsw(self, train_embeddings: np.ndarray, val_embeddings: np.ndarray):
        # Declaring index
        num_elements = self.total_dataset_size
        hnws_db = hnswlib.Index(space='cosine', dim=self.embedding_size) # possible options are l2, cosine or ip
        # Initializing index - the maximum number of elements should be known beforehand
        hnws_db.init_index(max_elements=num_elements, ef_construction=500, M=64)
        # Element insertion (can be called several times):
        hnws_db.add_items(train_embeddings)#, train_labels.cpu())
        # Controlling the recall by setting ef:
        hnws_db.set_ef(500) # ef should always be > k

        nearest_labels, nearest_distances = hnws_db.knn_query(val_embeddings, k=self.config['num_neighbors'])

        return nearest_distances, nearest_labels


    def create_and_search_index(self, train_embeddings: np.ndarray, val_embeddings: np.ndarray):
        k = self.config['num_neighbors']
        neighbors_model = NearestNeighbors(n_neighbors=k, metric='cosine')
        neighbors_model.fit(train_embeddings)
        distances, indices = neighbors_model.kneighbors(val_embeddings, k, return_distance=True)

        return distances, indices


    def create_val_targets_df(self, train_targets, val_image_names, val_targets):
        allowed_targets = np.unique(train_targets)
        val_targets_df = pd.DataFrame(np.stack([val_image_names, val_targets], axis=1), columns=["image", "target"])
        val_targets_df.loc[~val_targets_df.target.isin(allowed_targets), "target"] = "new_individual"

        return val_targets_df


    def create_distances_df(self, image_names, targets, D, I, stage):

        distances_df = []
        for i, image_name in tqdm(enumerate(image_names), desc=f"Creating {stage}_df"):
            target = targets[I[i]]
            distances = D[i]
            subset_preds = pd.DataFrame(np.stack([target, distances], axis=1), columns=["target", "distances"])
            subset_preds["image"] = image_name
            distances_df.append(subset_preds)

        distances_df = pd.concat(distances_df).reset_index(drop=True)
        distances_df = distances_df.groupby(["image", "target"]).distances.max().reset_index()
        distances_df = distances_df.sort_values("distances", ascending=False).reset_index(drop=True)

        return distances_df

    
    def get_best_threshold(self, val_targets_df, valid_df):
        best_th = 0
        best_cv = 0
        for th in [0.1 * x for x in range(11)]:
            all_preds = self.get_predictions(valid_df, threshold=th)

            cv = 0
            for i, row in val_targets_df.iterrows():
                target = row.target
                preds = all_preds[row.image]
                val_targets_df.loc[i, th] = self.map_per_image(target, preds)

            cv = val_targets_df[th].mean()

            print(f"th={th} cv={cv}")

            if cv > best_cv:
                best_th = th
                best_cv = cv

        print(f"best_th={best_th}")
        print(f"best_cv={best_cv}")

        # Adjustment: Since Public lb has nearly 10% 'new_individual' (be careful for private LB)
        val_targets_df["is_new_individual"] = val_targets_df.target == "new_individual"
        val_scores = val_targets_df.groupby("is_new_individual").mean().T
        val_scores["adjusted_cv"] = val_scores[True] * 0.1 + val_scores[False] * 0.9
        best_th = val_scores["adjusted_cv"].idxmax()
        print(f"best_th_adjusted={best_th}")

        return best_th, best_cv


    def map_per_image(self, label, predictions):
        try:
            return 1 / (predictions[:5].index(label) + 1)
        except ValueError:
            return 0.0


    def create_predictions_df(self, test_df: pd.DataFrame, best_th: float) -> pd.DataFrame:
        predictions = self.get_predictions(test_df, best_th)

        predictions = pd.Series(predictions).reset_index()
        predictions.columns = ["image", "predictions"]
        predictions["predictions"] = predictions["predictions"].apply(lambda x: " ".join(x))

        return predictions


    def get_predictions(self, df, threshold=0.2):
        sample_list = ["938b7e931166", "5bf17305f073", "7593d2aee842", "7362d7a01d00", "956562ff2888"]

        predictions = {}
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Creating predictions for threshold={threshold}"):
            if row.image in predictions:
                if len(predictions[row.image]) == 5:
                    continue
                predictions[row.image].append(row.target)
            elif row.distances > threshold:
                predictions[row.image] = [row.target, "new_individual"]
            else:
                predictions[row.image] = ["new_individual", row.target]

        for x in tqdm(predictions):
            if len(predictions[x]) < 5:
                remaining = [y for y in sample_list if y not in predictions]
                predictions[x] = predictions[x] + remaining
                predictions[x] = predictions[x][:5]

        return predictions

    
    def infer(self):

        self.load_eval_model()
        self.load_dataloaders()
        self.load_encoder()

        train_image_names, train_embeddings, train_targets = self.get_embeddings(self.train_loader, stage="train")
        val_image_names, val_embeddings, val_targets = self.get_embeddings(self.val_loader, stage="val")
        test_image_names, test_embeddings, test_targets = self.get_embeddings(self.test_loader, stage="test")

        D, I = self.create_and_search_index(train_embeddings, val_embeddings)  # noqa: E741
        print("Created index with train_embeddings")

        val_targets_df = self.create_val_targets_df(train_targets, val_image_names, val_targets)
        print(f"val_targets_df=\n{val_targets_df.head()}")

        val_df = self.create_distances_df(val_image_names, train_targets, D, I, "val")
        print(f"val_df=\n{val_df.head()}")

        best_th, best_cv = self.get_best_threshold(val_targets_df, val_df)
        print(f"val_targets_df=\n{val_targets_df.describe()}")

        train_embeddings = np.concatenate([train_embeddings, val_embeddings])
        train_targets = np.concatenate([train_targets, val_targets])
        print("Updated train_embeddings and train_targets with val data")

        D, I = self.create_and_search_index(train_embeddings, test_embeddings)  # noqa: E741
        print("Created index with train_embeddings")

        test_df = self.create_distances_df(test_image_names, train_targets, D, I, "test")
        print(f"test_df=\n{test_df.head()}")

        predictions = self.create_predictions_df(test_df, best_th)
        print(f"predictions.head()={predictions.head()}")
        predictions.to_csv(f'{self.title_run}_submission.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    predictor = Predictor(args.path)
    predictor.infer()