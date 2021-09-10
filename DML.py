from src.litmodels.dml import LitModel
from src.datamodules.cifar100_datamodule import DataModule

import pytorch_lightning as pl
import torch
import wandb

import torchvision.transforms as T

import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datapath", type=str, help='specify the path to cifar-100 dataset', required=True)
parser.add_argument("-s", "--save", type=str, help='specify where to save trained models', required=True)
parser.add_argument("-g", "--gpu", type=int, help='specify id of gpu to use; -1 denotes using all available gpus', default=-1, )
parser.add_argument("-e", "--epochs", type=int, help='number of training epochs', default=200)
parser.add_argument("-b", "--batch", type=int, help='batch size value', default=64)
parser.add_argument("-w", "--workers", type=int, help='number of workers to use for loading data', default=4)

args = parser.parse_args()

train_transforms = T.Compose([
                    T.RandomCrop(28, padding=4, padding_mode='reflect'),
                    T.RandomHorizontalFlip(),  # randomly flip image horizontally
                    T.ToTensor(),
                       ])

test_transforms = T.Compose([
                    T.ToTensor(),
                        ])


def train(dataset_path:str, save_dir:str, gpu_id:int, max_epochs:int, batch_size:int, num_workers:int, ):
    # datamodule & litmodel
    dm = DataModule(dataset_path, train_transform=train_transforms, test_transform=test_transforms,
                    batch_size=batch_size, num_workers=num_workers)
    litmodel = LitModel()

    # logger & callback
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    logger = pl.loggers.wandb.WandbLogger(project='distilled models')

    # trainer
    trainer = pl.Trainer(callbacks=[lr_monitor], logger=logger, 
                         gpus=gpu_id, max_epochs=max_epochs, 
                         val_check_interval=1., )

    # train & test
    trainer.fit(litmodel, dm)
    trainer.test(litmodel)
    
    # save trained
    torch.save(litmodel.resnet1.state_dict(), os.path.join(save_dir, 'dml_s1_resnet32.pth'))
    torch.save(litmodel.resnet2.state_dict(), os.path.join(save_dir, 'dml_s2_resnet32.pth'))

    wandb.finish()

if __name__=='__main__':
    train(args.datapath, args.save, args.gpu, args.epochs, args.batch, args.workers)
