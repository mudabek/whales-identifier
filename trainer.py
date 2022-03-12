# General imports
import os
import wandb
import numpy as np
import torch
from tqdm import tqdm
import copy
import pathlib

# Turn off wandb logging by default
os.environ['WANDB_MODE'] = 'offline'


class ModelTrainer:

    def __init__(self, model, dataloaders, criterion, optimizer, config):
        # Training related
        self.model = model
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = int(config['n_epochs'])
        self.device = torch.device(config['device'])
        self.model.to(self.device)
        self.save_last_model = config['save_last_model']

        # Model results stuff        
        self.best_epoch_loss = np.inf
        self.best_model_weights = None
        self.checkpoint = None  # last model and optimizer weights
        self.save_path = config["save_dir"]
        self.title_run = config["title_run"]

        # Logging to wandb
        if config['wandb_logging']:
            os.environ['WANDB_MODE'] = 'online'
        wandb.init(project="happy-whales", entity="obatek")
        wandb.config.update(config)


    def train_one_epoch(self):
        self.model.train()

        dataset_size = 0
        running_loss = 0.0
        
        for image_data, label in tqdm(self.train_loader):
            images = image_data.to(self.device, dtype=torch.float)
            labels = label.to(self.device, dtype=torch.long)

            batch_size = images.size(0)

            outputs = self.model(images, labels)
            loss = self.criterion(outputs, labels)  
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            wandb.log({"train_loss": loss})
            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size
            
            epoch_loss = running_loss / dataset_size

        return epoch_loss 


    def valid_one_epoch(self):
        self.model.eval()

        dataset_size = 0
        running_loss = 0.0

        for image_data, label in tqdm(self.val_loader):        
            images = image_data.to(self.device, dtype=torch.float)
            labels = label.to(self.device, dtype=torch.long)
            
            batch_size = images.size(0)

            outputs = self.model(images, labels)
            loss = self.criterion(outputs, labels)
            
            wandb.log({"val_loss": loss}) 
            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size
            
            epoch_loss = running_loss / dataset_size
        
        return epoch_loss

    
    def train_model(self):
        print('===> Training started')
        self.best_model_weights = copy.deepcopy(self.model.state_dict())
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"Epoch {epoch}/{self.num_epochs + 1}:")
            train_epoch_loss = self.train_one_epoch()
            val_epoch_loss = self.valid_one_epoch()

            wandb.log({"epoch_train_loss": train_epoch_loss})
            wandb.log({"epoch_valid_loss": val_epoch_loss})
            
            # deep copy the model
            if val_epoch_loss <= self.best_epoch_loss:
                print("===> Updating best model")
                self.best_epoch_loss = val_epoch_loss
                self.best_model_weights = copy.deepcopy(self.model.state_dict())

        if self.save_last_model:
            self.checkpoint = {'model_state_dict': copy.deepcopy(self.model.state_dict()),
                               'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict())}
        # load best model weights
        self.model.load_state_dict(self.best_model_weights)


    def save_results(self):
        print('===> Saving')
        path_to_dir = pathlib.Path(self.save_path)

        # Check if the directory exists:
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)

        # Save best model weights:
        torch.save(self.best_model_weights, path_to_dir / f'{self.title_run}_best_model_weights.pt')

        # Save last model weights (checkpoint):
        if self.save_last_model:
            torch.save(self.checkpoint, path_to_dir / f'{self.title_run}_last_model_checkpoint.tar')


    def load_model_weights(self, path_to_checkpoint):
        print('===> Loading')
        checkpoint = torch.load(path_to_checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])