"""Learned progress-to-hand head; no endpoint vector is supplied as an input."""
import torch
from torch import nn
class ProgressHand(nn.Module):
 def __init__(self,horizon=16,channels=12):
  super().__init__();self.horizon=horizon;self.channels=channels
  self.net=nn.Sequential(nn.Linear(65,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,horizon*channels))
 def forward(self,p):
  freq=torch.arange(1,33,device=p.device,dtype=p.dtype)[None]*torch.pi
  phase=p*freq
  return self.net(torch.cat([p,torch.sin(phase),torch.cos(phase)],-1)).reshape(-1,self.horizon,self.channels)
