import os
import glob

from skimage.io import imread
from torch.utils.data import Dataset, DataLoader

class CarvanaDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.image_dir = os.path.join(root_dir, split)
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        if split == "train":
            self.mask_dir = os.path.join(root_dir, split + "_masks")
            self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, "*.gif")))
        self.split = split
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.split == "train":
            image = imread(self.image_paths[idx])
            mask = imread(self.mask_paths[idx], as_gray=True)
            return image, mask
        if self.split == "test":
            image = imread(self.image_paths[idx])
            return image