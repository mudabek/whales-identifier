import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset


class HappyWhaleDataset(Dataset):

    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['encoded_id'].values
        self.transforms = transforms
        

    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return img, torch.tensor(label, dtype=torch.long)



class HappyWhalePredictionDataset(Dataset):

    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.image_ids = df['image'].values
        self.transforms = transforms
        

    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(image=img)["image"]


        image_id = self.image_ids[index]
            
        return img, image_id