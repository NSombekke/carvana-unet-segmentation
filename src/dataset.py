import os
import glob
import torch

from skimage.io import imread
from torch.utils.data import Dataset, DataLoader, random_split

class CarvanaDataset(Dataset):
    def __init__(self, root_dir, split="train", image_transform=None, mask_transform=None):
        assert split in ["train", "test"]
        self.image_dir = os.path.join(root_dir, split)
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        self.mask_dir = os.path.join(root_dir, split + "_masks")
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, "*.gif")))
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.split == "train":
            image = imread(self.image_paths[idx])
            mask = imread(self.mask_paths[idx], as_gray=True)
            print(image.dtype, mask.dtype)
            if self.image_transform and self.mask_transform:
                image, mask = self.image_transform(image), self.mask_transform(mask)
            return image, mask
        elif self.split == "test":
            image = imread(self.image_paths[idx])
            if self.image_transform:
                image = self.image_transform(image)
            return image
        
def get_dataloaders(root_dir, batch_size, num_workers, transforms, split_ratio=0.8):
    image_train_transform, mask_transform, image_test_transform = transforms['image'], transforms['mask'], transforms['test']
    train_ds = CarvanaDataset(root_dir, split="train", image_transform=image_train_transform, mask_transform=mask_transform)
    train_ds, val_ds = random_split(train_ds, [int(len(train_ds) * split_ratio), int(len(train_ds) * (1-split_ratio)) + 1])
    test_ds = CarvanaDataset(root_dir, split="test", image_transform=image_test_transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl

#====================================================================================================

def download_datasets(data_dir='../data', quiet=False):
    import zipfile
    import os
    from kaggle.api.kaggle_api_extended import KaggleApi
    # Make sure Kaggle public API is available 
    # ref.: https://www.kaggle.com/docs/api
    api = KaggleApi()
    api.authenticate()
    # Download files
    print("Downloading files...")
    api.competition_download_file('carvana-image-masking-challenge', file_name='metadata.csv.zip', path=data_dir, quiet=quiet)
    api.competition_download_file('carvana-image-masking-challenge', file_name='sample_submission.csv.zip', path=data_dir, quiet=quiet)
    api.competition_download_file('carvana-image-masking-challenge', file_name='test.zip', path=data_dir, quiet=quiet)
    api.competition_download_file('carvana-image-masking-challenge', file_name='train.zip', path=data_dir, quiet=quiet)
    api.competition_download_file('carvana-image-masking-challenge', file_name='train_masks.csv.zip', path=data_dir, quiet=quiet)
    api.competition_download_file('carvana-image-masking-challenge', file_name='train_masks.zip', path=data_dir, quiet=quiet)
    # Extract and remove zip files
    print("Extracting files...")
    for item in os.listdir(data_dir):
        if item.endswith('.zip'):
            file_name = os.path.join(data_dir, item)
            zip_ref = zipfile.ZipFile(file_name)
            zip_ref.extractall(data_dir)
            zip_ref.close()
            os.remove(file_name)
    print("Finished downloading dataset!")
    