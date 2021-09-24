import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
import numpy as np
import pandas as pd
import os

def read_image(path):
    img = Image.open(path)
    img = img.convert('RGB')
    return img

class CIFARDataset(Dataset):
    
    def __init__(self, img_paths, labels, transform=None):
        super().__init__()
        self.transform = transform
        self.img_paths = img_paths
        self.labels = labels
        
    def __len__(self):
        return self.img_paths.shape[0]
    
    def __getitem__(self, i):
        img = read_image(self.img_paths[i])
        label = self.labels[i]

        if self.transform:
            try: # albumentations
                img = self.transform(image=np.array(img))['image']
                img = np.transpose(img, (2, 0, 1)).astype('float32')
            except: # torchvision
                img = self.transform(img)
                
        return img, label

class DataModule(pl.LightningDataModule):
    
    def __init__(self, path, train_transform, test_transform, batch_size=32, num_workers=8, val_split=0.0):
        super().__init__()
        
        self.path = path
        self.val_split = val_split
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def prepare_data(self):
        def read(train):
            traincsv = 'train.csv' if train else 'test.csv'
            
            df = pd.read_csv(os.path.join(self.path, traincsv)).to_numpy()
            img_paths, labels = df[:, 0], df [:, 1]
            
            traindir = 'train/' if train else 'test/'
            img_paths = os.path.join(self.path, traindir) + img_paths
            
            return CIFARDataset(img_paths, labels, self.train_transform if train else self.test_transform)
        
        self.train = read(True)
        self.test = read(False)
        
    def setup(self, stage=None):
        if (stage is None or stage=='fit') and self.val_split>0:
            n = len(self.train)
            val = int(n*self.val_split)
            train = n - val
            self.train, self.val = random_split(self.train, [train, val])
            self.val.transform = self.test_transform
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        if self.val_split>0:
            return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        return None
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
