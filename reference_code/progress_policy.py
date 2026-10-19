"""Scalar-progress conditioned state-dominant policy, no absolute goal input."""
import torch
from torch import nn
class ProgressPolicy(nn.Module):
 def __init__(self,horizon=16,image_bound_rad=.001):
  super().__init__();self.horizon=horizon;self.image_bound_rad=image_bound_rad
  self.state=nn.Sequential(nn.Linear(8*26+1,384),nn.SiLU(),nn.Linear(384,384),nn.SiLU(),nn.Linear(384,256),nn.SiLU())
  self.arm=nn.Linear(256,horizon*14);self.hand=nn.Linear(256,horizon*12);self.events=nn.Linear(256,2)
  self.camera=nn.Sequential(nn.Conv2d(2,16,5,4,2),nn.SiLU(),nn.Conv2d(16,32,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d((3,4)),nn.Flatten(),nn.Linear(384,horizon*14))
 def infer(self,state,progress,image=None,scale=None):
  z=self.state(torch.cat([state.flatten(1),progress],-1));arm=self.arm(z).reshape(-1,self.horizon,14);hand=self.hand(z).reshape(-1,self.horizon,12);ev=self.events(z);pred=torch.cat([arm,hand],-1)
  if image is not None and scale is not None:
   corr=self.image_bound_rad*torch.tanh(self.camera(image).reshape(-1,self.horizon,14));pred=pred.clone();pred[:,:,:14]+=corr/scale[:14]
  return pred,ev
