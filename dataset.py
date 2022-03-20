import cv2
import torch
from torch.utils.data import Dataset


class HappyWhaleDataset(Dataset):

    def __init__(self, df, mode, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.image_ids = df['image'].values
        self.transforms = transforms
        self.mode = mode

        if mode != 'test':
            self.labels = df['encoded_id'].values
            self.invidual_ids = df['individual_id'].values

    
    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, index):
        sample = {}

        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
        
        sample['image'] = img
        sample['image_id'] = self.image_ids[index]

        if self.mode != 'test':
            sample['label'] = self.labels[index]
        else:
            sample['label'] = torch.tensor(1, dtype=torch.long)

        return sample
