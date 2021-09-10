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
    
    def __init__(self, ):
        super().__init__()
        
        self.automatic_optimization = False
        
        self.resnet1 = resnet32()
        self.resnet2 = resnet32()
        
        self.celoss = nn.CrossEntropyLoss()
        self.acc = Accuracy(num_classes=100, average='macro', compute_on_step=True)
        
    def configure_optimizers(self):
        opt = torch.optim.SGD([*self.resnet1.parameters(), *self.resnet2.parameters()], lr=.1, momentum=.9, nesterov=True)
        step = torch.optim.lr_scheduler.StepLR(opt, step_size=60, gamma=.1)
        return [opt], [step]
    
    def forward(self, x, optimize_first:bool=True):
        x1 = self.resnet1(x)
        x2 = self.resnet2(x)
        if not optimize_first:
            x1 = x1.detach()
        else:
            x2 = x2.detach()
        return x1, x2
    
    def training_step(self, batch, batch_id):
        x, y = batch

        opt = self.optimizers()
        
        # Optimize student 1
        x1, x2 = self(x)
        loss = self.celoss(x1, y) + kl_div(x2, x1)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        
        self.log('train_loss1', loss, prog_bar=True,)
        
        # Optimize student 2
        x1, x2 = self(x, False)
        loss = self.celoss(x2, y) + kl_div(x1, x2)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        
        self.log('train_loss2', loss, prog_bar=True,)
        
        if self.trainer.is_last_batch :
            self.lr_schedulers().step()
            
    def test_step(self, batch, *a):
        x, y = batch
        x1, x2 = self(x, True)
        x1, x2 = softmax(x1, dim=1), softmax(x2, dim=1)
        self.log('test_acc1', self.acc(x1, y), prog_bar=True)
        self.log('test_acc2', self.acc(x2, y), prog_bar=True)