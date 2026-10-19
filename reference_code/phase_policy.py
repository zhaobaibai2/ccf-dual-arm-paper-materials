"""State-dominant phase-conditioned policy, research-only."""
import torch
from torch import nn
class PhasePolicy(nn.Module):
 def __init__(self,nph,horizon=16,image_bound_rad=.001):
  super().__init__();self.nph=nph;self.horizon=horizon;self.image_bound_rad=image_bound_rad
  self.state_net=nn.Sequential(nn.Linear(8*26,384),nn.SiLU(),nn.Linear(384,384),nn.SiLU(),nn.Linear(384,256),nn.SiLU())
  self.phase=nn.Linear(256,nph);self.phase_emb=nn.Linear(nph,64)
  self.arm=nn.Linear(320,horizon*14);self.hand=nn.Linear(320,horizon*12);self.events=nn.Linear(320,2)
  self.camera=nn.Sequential(nn.Conv2d(2,16,5,4,2),nn.SiLU(),nn.Conv2d(16,32,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d((3,4)),nn.Flatten(),nn.Linear(384,horizon*14))
 def forward(self,state):
  z=self.state_net(state.flatten(1)); logits=self.phase(z);zp=torch.cat([z,self.phase_emb(torch.softmax(logits,-1))],-1)
  arm=self.arm(zp).reshape(-1,self.horizon,14);hand=self.hand(zp).reshape(-1,self.horizon,12);ev=self.events(zp)
  return torch.cat([arm,hand],-1),ev,logits
 def infer(self,state,image=None,scale=None,phase_probs=None):
  z=self.state_net(state.flatten(1)); logits=self.phase(z); probs=torch.softmax(logits,-1) if phase_probs is None else phase_probs
  zp=torch.cat([z,self.phase_emb(probs)],-1);arm=self.arm(zp).reshape(-1,self.horizon,14);hand=self.hand(zp).reshape(-1,self.horizon,12);ev=self.events(zp)
  pred=torch.cat([arm,hand],-1)
  if image is not None and scale is not None:
   corr=self.image_bound_rad*torch.tanh(self.camera(image).reshape(-1,self.horizon,14));pred=pred.clone();pred[:,:,:14]+=corr/scale[:14]
  return pred,ev,logits
