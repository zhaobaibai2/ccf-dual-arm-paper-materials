"""Hardware protocol adapter for frozen state-history/progress/CNN task1.
Trajectory states estimate scalar progress only. All actions come from the network.
"""
import argparse,copy,io,json,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
from .common import digest,npz_bytes
from .research_network import Network,MODELS
from .progress_deployment import MODEL,SHA

class MeasuredProgress:
    def __init__(self,q,active):
        self.q=np.asarray(q,float);self.mask=np.asarray(active,bool)
        self.radius=np.r_[np.full(14,.03),np.full(12,8.)]
        self.index=0
    def update(self,state):
        state=np.asarray(state,float)
        if state.shape!=(26,) or not np.isfinite(state).all():raise ValueError('finite measured state26 required')
        end=min(len(self.q),self.index+5)
        errors=np.max(np.abs(self.q[self.index:end]-state)[:,self.mask]/self.radius[self.mask],axis=1)
        reachable=np.flatnonzero(errors<=1.)
        if len(reachable):self.index+=int(reachable[-1])
        return self.index/(len(self.q)-1)


class Task1RecoveryProgress(MeasuredProgress):
    """Recover local overshoot using measured state, never elapsed time."""
    def update(self, state):
        before = self.index
        progress = super().update(state)
        if self.index != before:
            return progress
        states = self.q[before:min(len(self.q), before + 26)]
        # Check every hand channel, including inactive channels: do not cross
        # a grasp/release transition to recover arm progress.
        if np.any(np.abs(states[:, 14:] - self.q[before, 14:]) > 2.):
            return progress
        errors = np.max(np.abs(states - state)[:, self.mask] /
                        self.radius[self.mask], axis=1)
        nearest = int(errors.argmin())
        if nearest > 0 and errors[nearest] <= 1.:
            self.index += min(nearest, 4)
        return self.index / (len(self.q) - 1)

class Adapter:
    def __init__(self,device='cpu',model=None,expected_sha=None):
        model=model or MODEL;expected_sha=expected_sha or SHA
        if digest(model)!=expected_sha:raise ValueError('task1 progress checkpoint changed')
        self.network=Network('task1',model=model,device=device);p=self.network.payload
        dataset=Path(p['data_path'])/'dataset.npz'
        if digest(dataset)!=p['source_dataset_sha256']:raise ValueError('progress source dataset changed')
        with np.load(dataset,allow_pickle=False) as d:self.route=np.array(d['q'])
        if p.get('progress_route_override'):
            if digest(p['progress_route_override'])!=p['progress_route_sha256']:raise ValueError('progress route changed')
            self.route=np.load(p['progress_route_override'],allow_pickle=False)
        self.progress=Task1RecoveryProgress(self.route,p['data_meta']['active_mask'])
        d=copy.deepcopy(p['data_meta']);d.update(history=8,history_dt=.05,camera_selection=['top'],camera_views=1,offline_only=True)
        self.metadata=dict(schema='visual-cup-progress-cnn-adapter-v1',model_sha256=expected_sha,data_meta=d,
          model_config=dict(kind='bc',architecture='ProgressPolicy + learned progress arm/hand heads',channels=2,history=8,horizon=p['horizon'],with_state=True,goal_conditioned=False,camera_scale=1.,camera_views=1,hand_control_mode='continuous'),
          normalization=dict(center=p['center'],scale=p['scale']),
          progress_contract=dict(source='bounded measured-state local matching',dataset_sha256=p['source_dataset_sha256'],goal_input=False,trajectory_actions_sent=False,local_recovery=dict(version=1,window=25,max_advance=4,max_hand_change_raw=2,requires_measured_match=True)),
          physical_success_verified=False,adapter_hardware_tested=False,
          transition_fix=dict(learned_hand_head=bool(p.get('progress_hand_model')),learned_arm_head=bool(p.get('progress_arm_model')),
              index_target_raw=243,state_correction_bound_rad=.005,parent_sha256=p.get('parent_sha256')))
        self.lock=threading.Lock();self.end_since=None
    def predict(self,images,state):
        images=np.asarray(images,np.float32);state=np.asarray(state,np.float32)
        if images.shape!=(8,1,2,60,80) or state.shape!=(8,26):raise ValueError('requires images[8,1,2,60,80], state[8,26]')
        with self.lock:
            progress=self.progress.update(state[-1])
            answer=self.network.predict(state,images[-1,0],progress)
            # Learned event alone must not finish early; verify measured terminal pose.
            close=np.all(np.abs(state[-1]-self.route[-1])[self.progress.mask]<=self.progress.radius[self.progress.mask])
            if self.progress.index==len(self.route)-1 and close:
                if self.end_since is None:self.end_since=time.monotonic()
            else:self.end_since=None
            answer['events'][0]=float(self.end_since is not None and time.monotonic()-self.end_since>=.5)
            answer['progress_index']=np.asarray(self.progress.index)
            return answer

def serve(engine,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path!='/metadata':self.send_error(404);return
            b=json.dumps(engine.metadata).encode();self.send_response(200);self.end_headers();self.wfile.write(b)
        def do_POST(self):
            try:
                if self.path!='/predict':raise ValueError('unknown endpoint')
                n=int(self.headers.get('Content-Length',0))
                if not 0<n<=16*1024*1024:raise ValueError('invalid request size')
                with np.load(io.BytesIO(self.rfile.read(n)),allow_pickle=False) as d:b=npz_bytes(**engine.predict(d['images'],d['state']))
                self.send_response(200);self.end_headers();self.wfile.write(b)
            except Exception as e:self.send_error(400,str(e))
    print(engine.metadata['schema']+' ready',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8145);p.add_argument('--device',default='cpu');p.add_argument('--check',action='store_true');a=p.parse_args()
    engine=Adapter(a.device)
    if a.check:
        q=np.repeat(engine.route[:1],8,axis=0);images=np.zeros((8,1,2,60,80),np.float32)
        answer=engine.predict(images,q)
        assert answer['actions'].shape==(16,26) and np.isfinite(answer['actions']).all()
        print(json.dumps(dict(model_sha256=SHA,goal_input=False,network='ProgressPolicy transition v4',camera_scale=1.,actions_shape=list(answer['actions'].shape),hardware_actuated=False)))
    else:serve(engine,a.port)
if __name__=='__main__':main()
