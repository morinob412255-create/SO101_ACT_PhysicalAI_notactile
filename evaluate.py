"""Windows SO-101 ACT evaluator, refactored from the experiment's vision-only path.

No sensor driver, force values, private project directory, or hidden module is used.
S/F labels are supplied by the operator, not by an automatic success classifier.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path
import cv2
import numpy as np
import torch
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from vision import mask_external, apply_hand_controls

BASE = Path(__file__).resolve().parent
JOINTS = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
          'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
ALLOWED = {'observation.state', 'observation.images.external', 'observation.images.hand'}

def load_policy(checkpoint, device):
    policy = get_policy_class('act').from_pretrained(str(checkpoint)).to(device)
    if set(policy.config.input_features) != ALLOWED:
        raise ValueError('Only the three vision-only inputs are supported')
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={'device_processor': {'device': device}})
    return policy, pre, post

def infer(policy, pre, post, frame, device, task, robot_name='so101_follower'):
    batch = prepare_observation_for_inference(
        frame, torch.device(device), task=task, robot_type=robot_name)
    autocast = torch.autocast(device_type='cuda') if device.startswith('cuda') and policy.config.use_amp else nullcontext()
    with torch.inference_mode(), autocast:
        action = post(policy.select_action(pre(batch)))
    if not torch.isfinite(action).all():
        raise RuntimeError('Policy generated non-finite action')
    return action

def reset(policy, pre, post):
    policy.reset(); pre.reset(); post.reset()

def home(robot, path):
    payload = json.loads(path.read_text())
    target = payload['pose']
    if set(target) != set(JOINTS):
        raise ValueError('Home pose must specify all six joints')
    obs = robot.get_observation()
    start = {k: float(obs[k]) for k in JOINTS}
    for i in range(1, 251):
        t = time.perf_counter()
        robot.send_action({k: start[k] + i / 250 * (target[k] - start[k]) for k in JOINTS})
        time.sleep(max(0, .02 - (time.perf_counter() - t)))
    obs = robot.get_observation()
    if max(abs(float(obs[k]) - target[k]) for k in JOINTS) > payload.get('tolerance', 3):
        raise RuntimeError('Start pose is outside tolerance')

def failure(task):
    menus = {
        'stack': ['stage1/grasp', 'stage1/transport', 'stage1/placement', 'stage2/grasp', 'stage2/placement'],
        'prism': ['stage1/first_grasp', 'stage1/topple', 'stage2/regrasp', 'stage2/restore_upright'],
        'cup': ['initial_grasp_failure', 'nesting_failure', 'insufficient_nesting',
                'pair_lift_failure', 'post_lift_drop_or_separation', 'final_placement_failure']}
    for n, s in enumerate(menus[task], 1): print(f'{n}: {s}')
    v = input('Reason number (or free text): ').strip()
    if v.isdigit() and 1 <= int(v) <= len(menus[task]):
        v = menus[task][int(v) - 1]
    return v or 'unspecified'

def main():
    import msvcrt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['stack', 'prism', 'cup'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--follower-port', default='COM19')
    p.add_argument('--follower-id', default=None)
    p.add_argument('--external-camera', type=int, default=1)
    p.add_argument('--hand-camera', type=int, default=0)
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--device', default='cuda')
    p.add_argument('--home-pose', type=Path)
    p.add_argument('--dry-run', action='store_true', help='No policy or home commands; robot connection still occurs')
    a = p.parse_args()
    if a.trials < 1 or a.external_camera == a.hand_camera:
        raise ValueError('Use positive trials and two distinct camera IDs')
    if a.output.exists(): raise FileExistsError(a.output)
    profile = json.loads((BASE / f'configs/{a.task}.json').read_text())
    policy, pre, post = load_policy(a.checkpoint, a.device)
    ap, rap, rop = make_default_processors()
    cams = {k: OpenCVCameraConfig(index_or_path=i, width=640, height=480, fps=30, warmup_s=5)
            for k, i in [('external', a.external_camera), ('hand', a.hand_camera)]}
    robot = SOFollower(SO101FollowerConfig(port=a.follower_port, id=a.follower_id,
        cameras=cams, disable_torque_on_disconnect=False))
    homepath = a.home_pose or (BASE / 'assets' / profile['home'] if profile['home'] else None)
    rows = []
    try:
        robot.connect()
        controls = apply_hand_controls(robot.cameras['hand'], profile['hand_controls'])
        af = aggregate_pipeline_dataset_features(pipeline=ap,
            initial_features=create_initial_features(action=robot.action_features), use_videos=True)
        of = aggregate_pipeline_dataset_features(pipeline=rop,
            initial_features=create_initial_features(observation=robot.observation_features), use_videos=True)
        features = combine_feature_dicts(af, of)
        def observe():
            raw = robot.get_observation()
            frame = build_dataset_frame(features, rop(raw), prefix=OBS_STR)
            frame['observation.images.external'] = mask_external(
                frame['observation.images.external'], profile['mask'], BASE / 'assets/prism_fixed_roi.png')
            return raw, frame
        a.output.mkdir(parents=True)
        _, warm = observe()
        for k in ['external', 'hand']:
            cv2.imwrite(str(a.output / f'preview_{k}.png'), cv2.cvtColor(warm[f'observation.images.{k}'], cv2.COLOR_RGB2BGR))
        infer(policy, pre, post, warm, a.device, profile['task'], robot.name)
        reset(policy, pre, post)
        print('Inspect camera previews. Keys: H=home, R=start, S=success, F=fail, Q=quit.')
        if not a.dry_run and input('Type RUN to enable autonomous motion: ') != 'RUN': return
        for trial in range(1, a.trials + 1):
            ready = homepath is None or a.dry_run
            while True:
                key = msvcrt.getwch().lower()
                if key == 'q': return
                if key == 'h' and homepath and not a.dry_run:
                    home(robot, homepath); ready = True
                if key == 'r' and ready: break
            reset(policy, pre, post)
            td = a.output / f'trial_{trial:02d}'; td.mkdir()
            start = time.perf_counter(); count = 0; overruns = 0; video = None; result = None
            try:
                with (td / 'joint_positions.csv').open('w', newline='', encoding='utf8') as h:
                    jw = csv.writer(h); jw.writerow(['elapsed_s', 'sample_index', *JOINTS])
                    while result is None:
                        t = time.perf_counter(); elapsed = t - start
                        if elapsed >= profile['max_duration']: result = 'Fail_timeout'; break
                        if msvcrt.kbhit():
                            result = {'s': 'Success', 'f': 'Fail', 'q': 'Quit'}.get(msvcrt.getwch().lower())
                            if result: break
                        raw, frame = observe()
                        jw.writerow([elapsed, count, *[float(raw[k]) for k in JOINTS]])
                        image = frame['observation.images.external']
                        if video is None:
                            video = cv2.VideoWriter(str(td / 'external.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480))
                            if not video.isOpened(): raise RuntimeError('Video writer failed')
                        video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        action = infer(policy, pre, post, frame, a.device, profile['task'], robot.name)
                        act = make_robot_action(action, features)
                        if not a.dry_run: robot.send_action(rap((act, raw)))
                        count += 1
                        remain = 1/30 - (time.perf_counter() - t)
                        if remain > 0: time.sleep(remain)
                        else: overruns += 1
            finally:
                if video: video.release()
            reason = failure(a.task) if result == 'Fail' else ('total_timeout' if result == 'Fail_timeout' else '')
            meta = dict(trial=trial, result=result, failure_reason=reason,
                elapsed_s=time.perf_counter()-start, frames=count, overruns=overruns,
                mode='none', dry_run=a.dry_run, hand_camera_settings=controls)
            # Store trial elapsed before time spent in the operator's reason prompt.
            meta['elapsed_s'] = elapsed
            (td / 'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
            row = {k:meta[k] for k in ['trial','result','failure_reason','elapsed_s','frames','overruns']}
            rows.append(row)
            with (a.output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as h:
                w=csv.DictWriter(h,fieldnames=list(row));w.writeheader();w.writerows(rows)
            print(f'Trial {trial}: {result}. Policy commands paused; press H/R/Q.')
            if result == 'Quit': break
    finally:
        if robot.is_connected: robot.disconnect()

if __name__ == '__main__':
    main()
