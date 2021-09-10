import numpy as np
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm 
import os
from os.path import join


_HERE = os.path.split(os.path.abspath(__file__))[0]
_HERE = join(_HERE, 'dataset')

train = np.load(join(_HERE, 'train'), allow_pickle=True, encoding='bytes')
test = np.load(join(_HERE, 'test'), allow_pickle=True, encoding='bytes')


def save(root, csv, data):
    n = data[b'data'].shape[0]
    df = []
    for i in tqdm(range(n)):
        # extract each image
        img = data[b'data'][i]
        img = img.reshape(3, 32, 32)
        
        # extract class & super class
        fine_label = data[b'fine_labels'][i]
        coarse_label = data[b'coarse_labels'][i]
        
        # extract image name
        filename = data[b'filenames'][i]
        
        # save image
        img = Image.fromarray(img.transpose(1, 2, 0))
        img.save(root + os.sep + filename.decode("utf-8"))
        
        # save label
        df.append([filename.decode("utf-8"), fine_label, coarse_label])
        
    # save img name + class + super class in a csv file
    pd.DataFrame(df, columns=['image', 'fine_label', 'coarse_label']).to_csv(csv, index=False)
    
print("Setting up test data:")
save(join(_HERE, 'Test'), join(_HERE, 'test.csv'), test)

print("Setting up train data:")
save(join(_HERE, 'Train'), join(_HERE, 'train.csv'), train)
