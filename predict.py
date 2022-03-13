import argparse
import yaml
import pathlib
import joblib
import pandas as pd
from tqdm import tqdm

import torch

from sklearn.neighbors import NearestNeighbors

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import HappyWhalePredictionDataset, HappyWhaleDataset
import models


def main(args):
    path_to_config = pathlib.Path(args.path)
    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    # Hyperparameters
    device = torch.device(config['device'])
    num_neighb = config['num_neighbors']
    title_run = config['title_run']

    # Load label decoder
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = joblib.load(f)


    # Train and val data transforms:
    test_transforms = A.Compose([
        A.Resize(config['img_size'], config['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)

    # Train and test datasets
    test_df = pd.read_csv('submission_processed.csv')
    train_df = pd.read_csv(f"train_processed.csv")
    test_dataset = HappyWhalePredictionDataset(test_df, test_transforms)
    train_dataset = HappyWhaleDataset(train_df, test_transforms)

    # Model
    model = models.HappyWhaleModel(config)
    model.load_state_dict(torch.load(config['pred_model_path'], map_location=device))
    model.to(device)
    model.eval()

    # Get all the train embeddings for kNN training
    train_embeds = []
    train_labels = []
    print('===> Extracting embeddings')
    for image_data, label in tqdm(train_dataset):
        image_data = image_data.to(device, dtype=torch.float)
        image_emb = model.extract(image_data.unsqueeze(0))
        train_embeds.append(image_emb.squeeze(0).detach().cpu().numpy())
        train_labels.append(label)

        if len(train_embeds) > 10:
            break

    # Train nearest neighbors model
    print('===> Training NearestNeighbors')
    neighbors_model = NearestNeighbors(n_neighbors=num_neighb, metric='cosine')
    neighbors_model.fit(train_embeds)

    # Get nearest neighbors for testing set
    print("===> Getting predictions")
    image_ids = []
    predicted_ids = []
    for image_data, image_path in tqdm(test_dataset):
        # Get the embedding and n nearest neighbors
        image_data = image_data.to(device, dtype=torch.float)
        image_emb = model.extract(image_data.unsqueeze(0))
        image_emb = image_emb.detach().numpy()
        distances, neighb_idxs = neighbors_model.kneighbors(image_emb, num_neighb, return_distance=True)
        
        image_ids.append(image_path)

        # Create submission with the 5 most likely IDs
        distances = distances.flatten()
        neighb_idxs = neighb_idxs.flatten()
        sorted_idxs = distances.argsort()
        neighb_idxs = neighb_idxs[sorted_idxs]
        
        cur_pred = []
        cur_nn_idx = 0
        while (len(cur_pred) < 5):
            
            if neighb_idxs[cur_nn_idx] in cur_pred:
                cur_nn_idx = cur_nn_idx + 1
                continue

            # Decode the given label
            _, label = train_dataset[cur_nn_idx]
            decoded_label = label_encoder.inverse_transform([label.item()])[0]
            cur_pred.append(decoded_label)
            cur_nn_idx = cur_nn_idx + 1

        predicted_ids.append(' '.join(cur_pred))

        if len(image_ids) > 5:
            break

    
    data = {'image': image_ids, 'predictions': predicted_ids}
    submission_df = pd.DataFrame(data)
    
    assert submission_df.shape[0] == test_df.shape[0], "Number of prediction rows wrong"
    
    submission_df.to_csv(f'{title_run}_prediction.csv',index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    main(args)