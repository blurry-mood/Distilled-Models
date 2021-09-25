"""
Example:
    >>> python DML.py -d datasets/cifar-100/dataset -s models -g -1 -e 200 -b 64 -w 8 --nesterov --wandb
"""
from src.litmodels.dml import LitModel
from src.datamodules.cifar100_datamodule import DataModule

import pytorch_lightning as pl
import torch
import wandb

import torchvision.transforms as T

import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--wandb", help='Whether or not to log metrics to WandB', action='store_true')
parser.add_argument("--seed", type=int, help='Experiment seed, useful for results reproduction', default=1)

parser.add_argument("-d", "--datapath", type=str, help='Specify the path to cifar-100 dataset', required=True)
parser.add_argument("-s", "--save", type=str, help='Specify where to save trained models', required=True)
parser.add_argument("-g", "--gpu", type=int, help='Specify id of gpu to use; -1 denotes using all available gpus', default=-1, )
parser.add_argument("-e", "--epochs", type=int, help='Number of training epochs', default=200)
parser.add_argument("-b", "--batch", type=int, help='Batch size value', default=64)
parser.add_argument("-w", "--workers", type=int, help='Number of workers to use for loading data', default=4)
parser.add_argument("-v", "--val_split", type=float, help='Portion of training set to use for validation', default=0.1)

parser.add_argument("-l", "--lr", type=float, help='Initial learning rate for SGD', default=1e-1)
parser.add_argument("-m", "--momentum", type=float, help='Momentum of SGD', default=.9)
parser.add_argument("-W", "--weight_decay", type=float, help='Weight decay used for SGD', default=1e-3)
parser.add_argument("-n", "--nesterov", help='Whether or not to use Nesterov in SGD', action='store_true')

parser.add_argument("-S", "--step_size", type=int, help='Number of epochs to wait before reducing learning rate', default=60)
parser.add_argument("-G", "--gamma", type=float, help='Learning rate decay coefficient. After every `step_size` epochs, the learning rate is mutliplied by gamma', default=0.1)

args = parser.parse_args()

pl.seed_everything(args.seed)

train_transforms = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode='reflect'),  
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=15),  
        T.ToTensor(),  
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
                       ])

test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
                        ])


def train(args):
    # datamodule & litmodel
    dm = DataModule(args.datapath, train_transform=train_transforms, test_transform=test_transforms,
                    batch_size=args.batch, num_workers=args.workers, val_split=args.val_split )
    litmodel = LitModel(args.lr, args.momentum, args.weight_decay, args.step_size, args.gamma, args.nesterov)

    # logger & callback
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    trainer_args = dict(callbacks=[lr_monitor], gpus=args.gpu, max_epochs=args.epochs, val_check_interval=1.,)
    if args.wandb:
        logger = pl.loggers.wandb.WandbLogger(project='distilled models')
        trainer_args['logger'] = logger

    # trainer
    trainer = pl.Trainer(**trainer_args)

    # train & test
    trainer.fit(litmodel, dm)
    trainer.test(litmodel)
    
    # save trained
    torch.save(litmodel.resnet1.state_dict(), os.path.join(args.save, 'dml_s1_resnet32.pth'))
    torch.save(litmodel.resnet2.state_dict(), os.path.join(args.save, 'dml_s2_resnet32.pth'))

    wandb.finish()

if __name__=='__main__':
    train(args)
