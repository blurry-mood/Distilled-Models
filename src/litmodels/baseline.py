import torch
from torch import nn
from torch.nn.functional import softmax

from torchmetrics import Accuracy

import pytorch_lightning as pl

import wandb

from ..models.resnet_cifar import resnet32

torch.backends.cudnn.benchmark = True

class LitModel(pl.LightningModule):
    
    def __init__(self, lr, momentum, weight_decay, step_size, gamma, nesterov):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model = resnet32()
        
        self.celoss = nn.CrossEntropyLoss()
        self.top1 = Accuracy(num_classes=100, compute_on_step=True, top_k=1)
        self.top5 = Accuracy(num_classes=100, compute_on_step=True, top_k=5)
        
    def configure_optimizers(self):
        hp = self.hparams
        opt = torch.optim.SGD(self.model.parameters(), lr=hp.lr, momentum=hp.momentum, nesterov=hp.nesterov, weight_decay=hp.weight_decay)
        step = torch.optim.lr_scheduler.StepLR(opt, step_size=hp.step_size, gamma=hp.gamma, )
        return [opt], [step]
    
    def forward(self, x):
        return self.model(x)
    
    def _step(self, batch, split):
        x, y = batch
        y_hat = self(x)
        
        loss = self.celoss(y_hat, y)
        
        y_hat = softmax(y_hat, dim=1)
        top1 = self.top1(y_hat, y)
        top5 = self.top5(y_hat, y)
        
        self.log(f'{split}_loss', loss, prog_bar=True)
        self.log(f'{split}_top1', top1, prog_bar=True)
        self.log(f'{split}_top5', top5, prog_bar=True)
        
        self.top1.reset()
        self.top5.reset()
        
        return loss 
    
    def training_step(self, batch, batch_id):
        return self._step(batch, 'train')
    
    def validation_step(self, batch, batch_id):
        return self._step(batch, 'val')    
            
    def test_step(self, batch, batch_id):
        return self._step(batch, 'test')