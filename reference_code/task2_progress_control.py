"""Task2 phase/CNN with learned progress action head. No goal26 input."""
import argparse,hashlib,json,threading,time
from pathlib import Path
import numpy as np
import torch
from .phase_control import Adapter as PhaseAdapter,DATA,SHA as PARENT_SHA
from .progress_control import MeasuredProgress,serve
from .progress_hand import ProgressHand
from .recovery_progress import RecoveryProgress
from .common import digest
MODEL=Path('/home/autodrive/ccf/task12_routes_frozen_20261017/assets/task2_b_actions.pt')
def model_sha():
 return hashlib.sha256((PARENT_SHA+digest(MODEL)).encode()).hexdigest()
def first_grasp_weight(indices):
 indices=np.asarray(indices,float)
 return np.where((indices>=1511)&(indices<2488),np.minimum(np.clip((indices-1511)/30.,0.,1.),np.clip((2487-indices)/29.,0.,1.)),0.)

class Adapter(PhaseAdapter):
 def __init__(self,device='cpu'):
  if digest(MODEL)!='5d85578e84bbbabc75da5c949bd6acf769bba885a6b6682ae1321076e4fc7af1':raise ValueError('task2 progress checkpoint changed')
  super().__init__(device)
  p=torch.load(MODEL,map_location='cpu',weights_only=False)
  if digest(p['parent_model'])!=p['parent_sha256'] or digest(p['dataset'])!=p['dataset_sha256']:raise ValueError('task2 progress source changed')
  self.head=ProgressHand(channels=26).to(device);self.head.load_state_dict(p['state_dict']);self.head.eval();self.device=device
  self.center=np.asarray(self.network.payload['center']);self.scale=np.asarray(self.network.payload['scale'])
  if digest(p['progress_route'])!=p['progress_route_sha256']:raise ValueError('task2 progress route changed')
  self.route=np.load(p['progress_route']).copy()
  self.route[:,22]-=9.*first_grasp_weight(np.arange(len(self.route)))
  self.final=self.route[-1].copy();self.progress=RecoveryProgress(self.route,self.mask);self.lock=threading.Lock()
  self.metadata.update(model_sha256=model_sha(),schema='visual-cup-task2-phase-progress-cnn-v1',parent_model_sha256=PARENT_SHA,progress_action_sha256=digest(MODEL),progress_contract='monotonic local measured-state matching; scalar input; no goal26',completion_contract='measured final route position and stable dwell; no physical success claim')
  self.metadata['local_recovery']=dict(window_samples=25,max_advance_samples=4,max_hand_change_raw=2,requires_measured_match=True)
  self.metadata['label_revision']=p['label_revision']
  self.metadata['runtime_hand_override']=dict(version='task12-first-grasp-index205-v1',first_grasp=[205,220],second_grasp=[205,210],weights_changed=False,source='operator requested hand-only adjustment; not learned model output',progress_recovery_enabled=True)
  self.metadata['model_config']['architecture']='PhasePolicy CNN + learned scalar-progress action head'
 def predict(self,images,state):
  images=np.asarray(images,np.float32);state=np.asarray(state,np.float32)
  if images.shape!=(8,1,2,120,160) or state.shape!=(8,26):raise ValueError('invalid phase-progress input shape')
  with self.lock:
   self.progress.radius[22]=12. if (1511<=self.progress.index<2488 or 2796<=self.progress.index<4480) else 8.
   progress=self.progress.update(state[-1]);ans=self.network.predict(state,images[-1,0])
   with torch.no_grad():action=self.head(torch.tensor([[progress]],device=self.device,dtype=torch.float32))[0].cpu().numpy()*self.scale+self.center
   action[:,22]-=9.*first_grasp_weight(self.progress.index+10+np.arange(len(action)))
   action[:,:14]+=np.clip(ans['actions'][:,:14]-state[-1,:14],-.005,.005)
   ans['actions']=action.astype(np.float32);ans['progress']=np.asarray(progress);ans['progress_index']=np.asarray(self.progress.index)
   close=np.all(abs(state[-1]-self.final)[self.mask]<=self.progress.radius[self.mask])
   if self.progress.index==len(self.route)-1 and close:
    if self.since is None:self.since=time.monotonic()
   else:self.since=None
   ans['events'][0]=float(self.since is not None and time.monotonic()-self.since>=.5)
   return ans

def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--port',type=int,default=8146);p.add_argument('--device',default='cpu');a=p.parse_args();e=Adapter(a.device)
 if a.check:
  ans=e.predict(np.zeros((8,1,2,120,160),np.float32),np.repeat(e.route[:1],8,axis=0));assert np.isfinite(ans['actions']).all()
  print(json.dumps(dict(model_sha256=model_sha(),goal_input=False,actions_shape=list(ans['actions'].shape),hardware_actuated=False)))
 else:serve(e,a.port)
if __name__=='__main__':main()
