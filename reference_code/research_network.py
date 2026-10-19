"""Explicit inference interfaces for progress BC and learned phase BC.

This process only returns predictions. It never contacts a hardware owner.
Progress is an explicit caller input; no elapsed-time progress is invented.
"""
import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import torch
from .common import digest, npz_bytes
from .progress_policy import ProgressPolicy
from .phase_policy import PhasePolicy

MODELS = {
    'task1': '/home/autodrive/ccf/reports/training_task1_new_20261010/task1_progress_cnn_v2.pt',
    'task2': '/home/autodrive/ccf/task12_routes_frozen_20261017/assets/task2_b_phase.pt',
}

class Network:
    def __init__(self, task, model=None, device='cpu'):
        self.task = task
        path = model or MODELS[task]
        self.payload = torch.load(path, map_location='cpu', weights_only=False)
        p = self.payload
        expected = 'experimental-state-progress-v1' if task == 'task1' else 'experimental-state-phase-v1'
        if p['schema'] != expected or p['goal_conditioned']:
            raise ValueError('wrong checkpoint contract')
        self.device = device
        if task == 'task1':
            self.model = ProgressPolicy(p['horizon'], p['image_bound_rad'])
        else:
            self.model = PhasePolicy(**p['model_config'])
        self.model.load_state_dict(p['model'])
        self.model.to(device).eval()
        self.progress_hand=None
        if task=='task1' and 'progress_hand_model' in p:
            from .progress_hand import ProgressHand
            self.progress_hand=ProgressHand(p['horizon']).to(device)
            self.progress_hand.load_state_dict(p['progress_hand_model']);self.progress_hand.eval()
        self.progress_arm=None
        if task=='task1' and 'progress_arm_model' in p:
            from .progress_hand import ProgressHand
            self.progress_arm=ProgressHand(p['horizon'],channels=14).to(device)
            self.progress_arm.load_state_dict(p['progress_arm_model']);self.progress_arm.eval()
        self.center = np.asarray(p['center'], np.float32)
        self.scale = np.asarray(p['scale'], np.float32)
        self.lock = threading.Lock()
        self.image_height, self.image_width = ((60,80) if task == 'task1' else (120,160))
        self.metadata = dict(task=task, schema=expected, model_sha256=digest(path),
            model_path=path, goal_conditioned=False, camera_scale=p['camera_scale'],
            history_offsets_s=p['history_offsets_s'], phases=p.get('phases'),
            input_state_shape=[8,26], input_image_shape=[2,self.image_height,self.image_width],
            progress_required=task=='task1', hardware_actuated=False,
            real_robot_tested=False, note='Network predictions only; not connected to hardware.')

    def predict(self, state, image, progress=None):
        state = np.asarray(state, np.float32)
        image = np.asarray(image, np.float32)
        if state.shape != (8,26) or image.shape != (2,self.image_height,self.image_width):
            raise ValueError('requires state[8,26] and top depth image[2,60,80]')
        if not np.isfinite(state).all() or not np.isfinite(image).all():
            raise ValueError('nonfinite observation')
        with self.lock, torch.inference_mode():
            q = torch.as_tensor(((state-self.center)/self.scale)[None], device=self.device)
            im = torch.as_tensor(image[None], device=self.device)
            scale = torch.as_tensor(self.scale, device=self.device)
            if self.task == 'task1':
                if progress is None or np.asarray(progress).size != 1:
                    raise ValueError('task1 requires explicit scalar progress in [0,1]')
                value = float(np.asarray(progress).reshape(-1)[0])
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('progress outside [0,1]')
                pred, ev = self.model.infer(q, torch.tensor([[value]],device=self.device), im, scale)
                if self.progress_arm is not None:
                    bound=self.payload['progress_arm_config']['state_correction_bound_rad']/scale[:14]
                    correction=torch.clamp(pred[:,:,:14]-q[:,-1:,:14],-bound,bound)
                    pred=pred.clone();pred[:,:,:14]=self.progress_arm(torch.tensor([[value]],device=self.device))+correction
                if self.progress_hand is not None:
                    pred=pred.clone();pred[:,:,14:]=self.progress_hand(torch.tensor([[value]],device=self.device))
                answer = dict(progress=np.array(value, np.float32))
            else:
                pred, ev, logits = self.model.infer(q, im, scale)
                probs = torch.softmax(logits[0], -1).cpu().numpy()
                answer = dict(phase_probabilities=probs, phase_index=np.array(probs.argmax()),
                    phase_name=np.array(self.payload['phases'][probs.argmax()]))
            actions = pred[0].cpu().numpy()*self.scale+self.center
            if not np.isfinite(actions).all():
                raise ValueError('nonfinite prediction')
            answer.update(actions=actions, events=torch.sigmoid(ev[0]).cpu().numpy())
            return answer

def serve(engine, host, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            if self.path != '/metadata': self.send_error(404); return
            data = json.dumps(engine.metadata).encode()
            self.send_response(200); self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)
        def do_POST(self):
            try:
                if self.path != '/predict': raise ValueError('unknown endpoint')
                size = int(self.headers.get('Content-Length',0))
                if not 0 < size <= 1024*1024: raise ValueError('invalid request size')
                with np.load(io.BytesIO(self.rfile.read(size)), allow_pickle=False) as obs:
                    data = npz_bytes(**engine.predict(obs['state'],obs['image'],
                        obs['progress'] if 'progress' in obs.files else None))
                self.send_response(200); self.send_header('Content-Length', str(len(data)))
                self.end_headers(); self.wfile.write(data)
            except (ValueError, KeyError) as error:
                self.send_error(400, str(error))
    print(f'{engine.task} network ready: http://{host}:{port}', flush=True)
    ThreadingHTTPServer((host,port),Handler).serve_forever()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=MODELS, required=True)
    p.add_argument('--model'); p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port',type=int); p.add_argument('--device',default='cpu')
    p.add_argument('--input',help='offline NPZ with state, image, and task1 progress')
    p.add_argument('--output',help='prediction NPZ for offline input')
    a=p.parse_args(); engine=Network(a.task,a.model,a.device)
    if a.input:
        with np.load(a.input,allow_pickle=False) as obs:
            result=engine.predict(obs['state'],obs['image'],obs['progress'] if 'progress' in obs.files else None)
        if a.output: np.savez_compressed(a.output,**result)
        print(json.dumps({'task':a.task,'actions_shape':list(result['actions'].shape),
            'phase_name':str(result['phase_name']) if 'phase_name' in result else None,
            'hardware_actuated':False}))
    else: serve(engine,a.host,a.port or (8141 if a.task=='task1' else 8142))

if __name__=='__main__': main()
