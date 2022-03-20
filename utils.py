import yaml
import pathlib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import joblib
from dataset import HappyWhaleDataset
from torch.utils.data import DataLoader


def process_test_dataframe():
    # Get directories information from config
    path_to_config = pathlib.Path('config.yaml')
    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    ROOT_DIR = config['root_dir']

    # Helper function to add filepath column
    def get_file_path(id):
        return f"{ROOT_DIR}/test_images/{id}"

    # Add filepath to dataframes
    test_df = pd.read_csv(f"{ROOT_DIR}/sample_submission.csv")
    test_df['file_path'] = test_df['image'].apply(get_file_path)

    # Export final processed dataframe
    test_df.to_csv(f"submission_processed.csv")



def preprocess_train_dataframe(config):

    ROOT_DIR = config['root_dir']

    # Helper function to add filepath column
    def get_file_path(id):
        return f"{ROOT_DIR}/train_images/{id}"

    # Add filepath to dataframes
    train_df = pd.read_csv(f"{ROOT_DIR}/train.csv")
    train_df['file_path'] = train_df['image'].apply(get_file_path)

    # Encode fish IDs
    encoder = LabelEncoder()
    train_df['encoded_id'] = encoder.fit_transform(train_df['individual_id'])

    # Create folds for validation
    skf = StratifiedKFold(n_splits=config['n_fold'])

    for fold, ( _, val_) in enumerate(skf.split(X=train_df, y=train_df.encoded_id)):
        train_df.loc[val_ , "kfold"] = fold

    # Export final processed dataframe
    mode = config['mode']
    train_df.to_csv(f"train_processed_{mode}.csv")

    # Save the encoder
    with open(f"label_encoder_{mode}.pkl", "wb") as fp:
        joblib.dump(encoder, fp)



def prepare_loaders(data_transforms, config):
    fold = int(config['fold'])
    train_batch_size = int(config['train_batch_size'])
    val_batch_size = int(config['val_batch_size'])
    test_batch_size = int(config['test_batch_size'])
    num_workers = int(config['num_workers'])
    data_mode = config['mode']

    # Split dataframe into train and validation based on fold
    df = pd.read_csv(f'train_processed_{data_mode}.csv')
    df_test = pd.read_csv('submission_processed.csv')

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # Create dataloaders for training
    train_dataset = HappyWhaleDataset(df_train, mode='train', transforms=data_transforms["train"])
    valid_dataset = HappyWhaleDataset(df_valid, mode='valid', transforms=data_transforms["valid"])
    test_dataset = HappyWhaleDataset(df_test, mode='test', transforms=data_transforms["valid"])

    total_size = len(train_dataset) + len(valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=num_workers,
                              shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, val_batch_size, num_workers=num_workers, 
                              shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, test_batch_size, num_workers=num_workers, 
                              shuffle=False, pin_memory=True)
    
    return {"train": train_loader, "val": valid_loader, "test": test_loader, "total_size": total_size}