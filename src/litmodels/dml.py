import torch
from torch import nn
from torch.nn.functional import softmax,  log_softmax

from torchmetrics import Accuracy

import pytorch_lightning as pl

import wandb

from ..models.resnet_cifar import resnet32

torch.backends.cudnn.benchmark = True

def kl_div(x, y):
    px = softmax(x, dim=1)
    lpx, lpy = log_softmax(x, dim=1), log_softmax(y, dim=1)
    return (px*(lpx-lpy)).mean()

class LitModel(pl.LightningModule):
    
    def __init__(self, lr, momentum, weight_decay, step_size, gamma, nesterov):
        super().__init__()
        
        self.automatic_optimization = False
        self.save_hyperparameters()
        
        self.resnet1 = resnet32()
        self.resnet2 = resnet32()
        
        self.celoss = nn.CrossEntropyLoss()
        self.top1 = Accuracy(num_classes=100, compute_on_step=True, top_k=1)
        self.top5 = Accuracy(num_classes=100, compute_on_step=True, top_k=5)
        
    def configure_optimizers(self):
        hp = self.hparams
        
        opt1 = torch.optim.SGD(self.resnet1.parameters(), lr=hp.lr, momentum=hp.momentum, nesterov=hp.nesterov, weight_decay=hp.weight_decay)
        opt2 = torch.optim.SGD(self.resnet2.parameters(), lr=hp.lr, momentum=hp.momentum, nesterov=hp.nesterov, weight_decay=hp.weight_decay)
        step1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=hp.step_size, gamma=hp.gamma, )
        step2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=hp.step_size, gamma=hp.gamma, )
        return [opt1, opt2], [step1, step2]
    
    def forward(self, x):
        x1 = self.resnet1(x)
        x2 = self.resnet2(x)
        return x1, x2
    
    def training_step(self, batch, batch_id):
        x, y = batch

        opt1, opt2 = self.optimizers()
        
        # Optimize student 1
        x1, x2 = self(x)
        loss = self.celoss(x1, y) + kl_div(x2, x1)
        opt1.zero_grad()
        self.manual_backward(loss)
        opt1.step()
        
        self.log('train_loss1', loss, prog_bar=True,)
        
        # Optimize student 2
        x1, x2 = self(x)
        loss = self.celoss(x2, y) + kl_div(x1, x2)
        opt2.zero_grad()
        self.manual_backward(loss)
        opt2.step()
        
        self.log('train_loss2', loss, prog_bar=True,)
        
        if self.trainer.is_last_batch :
            self.lr_schedulers()[0].step()
            self.lr_schedulers()[1].step()
    
    def _step(self, batch, split):
        x, y = batch
        x1, x2 = self(x)
        
        loss1 = self.celoss(x1, y)
        loss2 = self.celoss(x2, y)

        x1, x2 = softmax(x1, dim=1), softmax(x2, dim=1)
        
        top11, top51 = self.top1(x1, y), self.top5(x1, y)
        self.top1.reset()
        self.top5.reset()
        
        top12, top52 = self.top1(x2, y), self.top5(x2, y)
        self.top1.reset()
        self.top5.reset()
        
        self.log(f'{split}_loss1', loss1, prog_bar=True)
        self.log(f'{split}_loss2', loss2, prog_bar=True)
        
        self.log(f'{split}_top1_1', top11, prog_bar=True)
        self.log(f'{split}_top1_2', top12, prog_bar=True)
        self.log(f'{split}_top5_1', top51, prog_bar=True)
        self.log(f'{split}_top5_2', top52, prog_bar=True)

    def test_step(self, batch, *a):
        self._step(batch, 'test')
        

    def validation_step(self, batch, *a):
        self._step(batch, 'val')        