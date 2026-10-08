"""Train a vision-only ACT using the recorded experiment hyperparameters."""
import argparse
import json
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['stack', 'prism', 'cup'], required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--repo-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--print-only', action='store_true')
    a = p.parse_args()
    profile = json.loads((BASE / f'configs/{a.task}.json').read_text())
    info = json.loads((a.dataset / 'meta/info.json').read_text())
    allowed = {'observation.state', 'observation.images.external', 'observation.images.hand'}
    actual = {k for k in info['features'] if k.startswith('observation.')}
    if actual != allowed:
        raise ValueError(f'Training expects vision-only features: {actual}')
    if a.output.exists():
        raise FileExistsError(a.output)
    cmd = ['lerobot-train', f'--dataset.root={a.dataset.resolve()}',
           f'--dataset.repo_id={a.repo_id}', f'--dataset.eval_split={profile["eval_split"]}',
           '--policy.type=act', '--policy.device=cuda', '--policy.push_to_hub=false',
           '--policy.chunk_size=100', '--policy.n_action_steps=100',
           '--policy.optimizer_lr=0.00001', '--policy.optimizer_lr_backbone=0.00001',
           f'--steps={profile["steps"]}', '--batch_size=4', '--seed=1000',
           '--num_workers=4', '--save_freq=2500', '--log_freq=500',
           f'--eval_steps={2500 if a.task == "cup" else 0}',
           f'--env_eval_freq={20000 if a.task == "cup" else 0}',
           '--wandb.enable=false', f'--output_dir={a.output.resolve()}',
           f'--job_name=so101_{a.task}_vision_only']
    print(subprocess.list2cmdline(cmd))
    if not a.print_only:
        log = a.output.with_suffix('.train.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w', encoding='utf8') as h:
            h.write('command=' + subprocess.list2cmdline(cmd) + '\n'); h.flush()
            result = subprocess.run(cmd, stdout=h, stderr=subprocess.STDOUT, check=False)
            h.write(f'\nexit_code={result.returncode}\n')
        if result.returncode:
            raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
