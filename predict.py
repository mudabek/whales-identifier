import argparse
import yaml
import gc
import pathlib
import joblib
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

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
    test_batch_size = config['test_batch_size']
    num_workers = config['num_workers']

    # Check device
    print(f'Training on {device}')

    # Load label decoder
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = joblib.load(f)

    # Data transforms:
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
    test_loader = DataLoader(test_dataset, test_batch_size, num_workers=num_workers, 
                             shuffle=False, pin_memory=True)
    train_loader = DataLoader(train_dataset, test_batch_size, num_workers=num_workers, 
                              shuffle=False, pin_memory=True)

    # Model
    model = models.HappyWhaleModel(config)
    model.load_state_dict(torch.load(config['pred_model_path'], map_location=device))
    model.to(device)
    model.eval()

    # Get all the train embeddings for kNN training
    train_embeds = torch.FloatTensor().to('cpu')
    train_labels = torch.LongTensor().to('cpu')
    print('===> Extracting embeddings')
    for image_data, labels in tqdm(train_loader):
        labels = labels.detach().cpu()
        image_data = image_data.to(device, dtype=torch.float)

        image_emb = model.extract(image_data).squeeze(0).detach().cpu()

        train_embeds = torch.cat((train_embeds,  image_emb), 0)
        train_labels = torch.cat((train_labels, labels), 0)

    gc.collect()

    # Train nearest neighbors model
    print('===> Gathering NearestNeighbors')
    neighbors_model = NearestNeighbors(n_neighbors=num_neighb, metric='cosine')
    neighbors_model.fit(train_embeds.numpy())


    # Get nearest neighbors for testing set
    print("===> Getting predictions")
    image_ids = []
    predicted_ids = []
    for image_data, image_path in tqdm(test_loader):
        # Get the embedding and n nearest neighbors
        image_data = image_data.to(device, dtype=torch.float)
        image_emb = model.extract(image_data).squeeze(0).detach().cpu()
        distances, neighb_idxs = neighbors_model.kneighbors(image_emb, num_neighb, return_distance=True)
        
        image_ids = image_ids + list(image_path)

        # Create submission with the 5 most likely IDs
        for i in range(len(distances)):
            cur_distance = distances[i]
            cur_idxs = neighb_idxs[i]

            sorted_idxs = cur_distance.argsort()
            cur_idxs = cur_idxs[sorted_idxs]
            
            cur_pred = []
            iter = 0
            while (len(cur_pred) < 5):
                # Skip if ID already in the predicted list
                if cur_idxs[iter] in cur_pred:
                    iter = iter + 1
                    continue

                # Decode the given label
                _, label = train_dataset[cur_idxs[iter]]
                decoded_label = label_encoder.inverse_transform([label.item()])[0]
                cur_pred.append(decoded_label)
                iter = iter + 1

            predicted_ids.append(' '.join(cur_pred))

    # Export the final submission file
    submission_data = {'image': image_ids, 'predictions': predicted_ids}
    submission_df = pd.DataFrame(submission_data)
    assert submission_df.shape[0] == test_df.shape[0], "Number of predicted rows wrong"
    submission_df.to_csv(f'{title_run}_prediction.csv',index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    main(args)