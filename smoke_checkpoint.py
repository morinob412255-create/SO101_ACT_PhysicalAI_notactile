"""Load a checkpoint and run one synthetic inference without a robot connection."""
import argparse
from pathlib import Path
import numpy as np
from evaluate import load_policy,infer
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args()
    policy,pre,post=load_policy(a.checkpoint,a.device)
    frame={'observation.state':np.zeros(6,np.float32),
           'observation.images.external':np.full((480,640,3),96,np.uint8),
           'observation.images.hand':np.full((480,640,3),96,np.uint8)}
    action=infer(policy,pre,post,frame,a.device,'pick_and_place')
    assert tuple(action.shape)==(1,6),action.shape
    print('PASS: vision-only inputs and finite six-joint action. No robot connection.')
if __name__=='__main__':main()
