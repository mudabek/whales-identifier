import argparse
import yaml
import pathlib
import numpy as np

import torch
import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensorV2

import trainer
import models
from utils import prepare_loaders


def main(args):
    path_to_config = pathlib.Path(args.path)
    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    # Read config:
    lr = float(config['lr'])

    # Train and val data transforms:
    data_transforms = {
        "train": A.Compose([
            A.Resize(config['img_size'], config['img_size']),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ),
            ToTensorV2()], p=1.),
        
        "valid": A.Compose([
            A.Resize(config['img_size'], config['img_size']),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ),
            ToTensorV2()], p=1.)
    }

    # Datasets:
    dataloaders = prepare_loaders(data_transforms, config)

    # Model
    model = models.HappyWhaleModel(config)

    # Training things
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    model_trainer = trainer.ModelTrainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
    )

    model_trainer.train_model()
    model_trainer.save_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    main(args)