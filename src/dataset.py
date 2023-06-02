import os
import glob

from skimage.io import imread
from torch.utils.data import Dataset, DataLoader, random_split

class CarvanaDataset(Dataset):
    def __init__(self, root_dir, split="train", split_ratio=0.8, image_transform=None, mask_transform=None):
        assert split in ["train", "val", "test"]
        self.image_dir = os.path.join(root_dir, split)
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        if split in ["train", "val"]:
            self.mask_dir = os.path.join(root_dir, split + "_masks")
            self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, "*.gif")))
            if split == "train":
                self.image_paths = self.image_paths[:int(split_ratio * len(self.image_paths))]
                self.mask_paths = self.mask_paths[:int(split_ratio * len(self.mask_paths))]
            elif split == "val":
                self.image_paths = self.image_paths[int(split_ratio * len(self.image_paths)):]
                self.mask_paths = self.mask_paths[int(split_ratio * len(self.mask_paths)):]
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.split in ["train", "val"]:
            image = imread(self.image_paths[idx])
            mask = imread(self.mask_paths[idx], as_gray=True)
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
    train_ds = CarvanaDataset(root_dir, split="train", split_ratio=split_ratio, image_transform=image_train_transform, mask_transform=mask_transform)
    val_ds = CarvanaDataset(root_dir, split="val", split_ratio=split_ratio, image_transform=image_train_transform, mask_transform=mask_transform)
    test_ds = CarvanaDataset(root_dir, split="test", image_transform=image_test_transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl    