import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np
import pandas as pd

def encode_labels(labels):
        le = LabelEncoder()
        encoded = le.fit_transform(labels)
        return le, encoded
    
def read_image(path, w, h):
    img = Image.open(path)
    img = img.resize((w, h))
    img = img.convert('RGB')
    return np.array(img)

    
class ImageNetDataset(Dataset):
    
    def __init__(self, width, height, img_paths, labels):
        super().__init__()
        self.width, self.height = width, height
        self.img_paths = img_paths
        self.encoder, self.labels = encode_labels(labels)
        
    def __len__(self):
        return self.img_paths.shape[0]
    
    def __getitem__(self, i):
        img = read_image(self.img_paths[i], self.width, self.height) 
        label = self.labels[i]
        img = np.transpose(img, (2, 0, 1)).astype('float32')/255
        return img, label
    
class DataModule(pl.LightningDataModule):
    
    def __init__(self, path, n, width, height, batch_size=32, num_workers=8, split={'train':.8, 'test':.1, 'val':.1}):
        super().__init__()
        
        self.path = path
        self.n = n
        self.width, self.height = width, height
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        
    def prepare_data(self):
        df = pd.read_csv(self.path).to_numpy()[:self.n]
        img_paths, labels = df[:, 0], df [:, 1]
        self.dataset = ImageNetDataset(self.width, self.height, img_paths, labels)
        
    def setup(self, stage=None):
        if stage=='fit' or stage is None:
            n = len(self.dataset)
            l = [self.split['val'], self.split['test']]
            val, test = map(lambda x: int(x*n), l)
            train = n - (val + test)
            self.train, self.val, self.test = random_split(self.dataset, [train, val, test], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
