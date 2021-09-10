import torch

from torch import nn
from torch.nn.functional import softmax,  log_softmax
from torchmetrics import Accuracy

from ..models.resnet_cifar import resnet32

import pytorch_lightning as pl

import wandb

torch.backends.cudnn.benchmark = True


class Attention(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        
        self.linear = nn.Linear(input_dim, input_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        xa = self.linear(x)
        b = xa @ x.transpose(-1, -2)
        c = self.softmax(b)
        y = c @ x
        return y
    
    
def kl_div(x, y):
    return (x*(x/y).log()).mean()

class LitModel(pl.LightningModule):
    
    def __init__(self, ):
        super().__init__()
        
        self.student1 = resnet32()
        self.student2 = resnet32()
        self.student3 = resnet32()
        self.leader = resnet32()
        
        self.mha = Attention(input_dim=100)
        
        self.T = 3.0
        
        self.celoss = nn.CrossEntropyLoss()
        self.acc = Accuracy(num_classes=100, average='macro', compute_on_step=True)
        
    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=.9, nesterov=True, weight_decay=5e-4)
        step = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[150, 255], gamma=.1)
        return [opt], [step]
    
    def forward(self, x, optimize_first:bool=True):
        x1 = self.student1(x)
        x2 = self.student2(x)
        x3 = self.student3(x)
        xl = self.leader(x)
        return x1, x2, x3, xl
    
    def training_step(self, batch, batch_id):
        x, y = batch        
        xs = self(x)
        
        # GT loss
        loss = [self.celoss(_x, y) for _x in xs]
        loss = torch.stack(loss, dim=0).sum()
        
        # peers loss
        t1, t2, t3, tl = [softmax(_x/self.T, dim=1) for _x in xs]
        peers = torch.stack((t1, t2, t3), dim=1)
        mha_peers = self.mha(peers)

        loss += self.T * kl_div(mha_peers, peers)
        
        # leader loss
        mean = peers.mean(dim=1)
        loss += self.T * kl_div(mean, tl)
        
        assert loss.item() == loss.item()
        
        # logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.acc(tl, y), prog_bar=True)
        
        return loss 
    
    def test_step(self, batch, *a):
        x, y = batch        
        xs = self(x)
        
        # GT loss
        loss = [self.celoss(_x, y) for _x in xs]
        loss = sum(loss)
        
        *t, tl = [softmax(_x, dim=1) for _x in xs]

        # logging
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.acc(tl, y), prog_bar=True)
        
        return loss 