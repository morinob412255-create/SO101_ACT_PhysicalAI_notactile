"""Record new vision-only demonstrations with the upstream LeRobot CLI."""
import argparse,json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['stack','prism','cup'],required=True)
    p.add_argument('--repo-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--follower-port',default='COM19')
    p.add_argument('--leader-port',default='COM18')
    p.add_argument('--external-camera',type=int,default=1)
    p.add_argument('--hand-camera',type=int,default=0)
    p.add_argument('--print-only',action='store_true')
    a=p.parse_args();profile=json.loads((BASE/f'configs/{a.task}.json').read_text())
    cameras={k:dict(type='opencv',index_or_path=i,width=640,height=480,fps=30)
             for k,i in [('external',a.external_camera),('hand',a.hand_camera)]}
    cmd=['lerobot-record','--robot.type=so101_follower',f'--robot.port={a.follower_port}',
         '--robot.id=so101_follower','--teleop.type=so101_leader',
         f'--teleop.port={a.leader_port}','--teleop.id=so101_leader',
         '--robot.cameras='+json.dumps(cameras,separators=(',',':')),
         f'--dataset.repo_id={a.repo_id}',f'--dataset.root={a.output.resolve()}',
         f'--dataset.single_task={profile["task"]}',
         f'--dataset.num_episodes={profile["episode_count"]}',
         f'--dataset.episode_time_s={20 if a.task=="cup" else 30}',
         '--dataset.reset_time_s=3','--dataset.push_to_hub=false','--display_data=false']
    print(subprocess.list2cmdline(cmd))
    if not a.print_only:subprocess.run(cmd,check=True)
if __name__=='__main__':main()
