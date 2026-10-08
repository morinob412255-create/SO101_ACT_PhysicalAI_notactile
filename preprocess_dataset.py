"""Copy a vision-only LeRobot v3 dataset and mask External videos.

Joint/action tables and Hand videos are kept unchanged. Source data is untouched.
The three tasks use ImageNet normalization for images, as recorded in configs.
External metadata statistics are recomputed from the processed frames.
"""
import argparse,json,shutil
from pathlib import Path
import cv2,numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import get_feature_stats, aggregate_stats
from vision import mask_external
BASE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['stack','prism','cup'],required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.source.resolve() in a.output.resolve().parents:
        raise ValueError('Output must be outside the source dataset')
    info=json.loads((a.source/'meta/info.json').read_text())
    actual={k for k in info['features'] if k.startswith('observation.')}
    if actual != {'observation.state','observation.images.external','observation.images.hand'}:
        raise ValueError('Source must contain only vision and joint state observations')
    profile=json.loads((BASE/f'configs/{a.task}.json').read_text())
    # Metadata is updated only after all videos are written successfully.
    shutil.copytree(a.source,a.output)
    video_key='observation.images.external';prefix='videos/'+video_key
    records=[];meta_files=sorted((a.output/'meta/episodes').rglob('*.parquet'))
    for mf in meta_files:
        records.extend(pq.read_table(mf).to_pylist())
    sample_banks={int(e['episode_index']):[] for e in records}
    files=sorted((a.output/'videos'/video_key).rglob('*.mp4'))
    if not files:raise ValueError('External video files not found')
    for file in files:
        cap=cv2.VideoCapture(str(file));fps=cap.get(cv2.CAP_PROP_FPS)
        temp=file.with_name(file.stem+'_masked.mp4')
        writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(640,480))
        if not cap.isOpened() or not writer.isOpened():raise RuntimeError(file)
        relevant=[]
        for e in records:
            rel=info['video_path'].format(video_key=video_key,chunk_index=e[prefix+'/chunk_index'],file_index=e[prefix+'/file_index'])
            if (a.output/rel).resolve()==file.resolve():relevant.append(e)
        count=0
        try:
            while True:
                ok,bgr=cap.read()
                if not ok:break
                rgb=mask_external(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),profile['mask'],BASE/'assets/prism_fixed_roi.png')
                writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                if count%5==0:
                    ts=count/fps
                    for e in relevant:
                        if e[prefix+'/from_timestamp']<=ts<e[prefix+'/to_timestamp']:
                            sample_banks[int(e['episode_index'])].append(rgb[::8,::8].transpose(2,0,1)/255.)
                count+=1
        finally:cap.release();writer.release()
        if count==0:raise ValueError('Empty video')
        temp.replace(file)
    image_stats=[]
    for e in records:
        images=np.asarray(sample_banks[int(e['episode_index'])],dtype=np.float32)
        if len(images)==0:raise ValueError('Episode has no image samples')
        s=get_feature_stats(images,axis=(0,2,3),keepdims=True)
        # LeRobot image stats have shape Cx1x1, with count independent of axes.
        for k in list(s):
            if k!='count':s[k]=np.squeeze(s[k],axis=0)
        s['count']=np.array([len(images)])
        image_stats.append({video_key:s})
        for k,v in s.items():e[f'stats/{video_key}/{k}']=v.tolist()
    offset=0
    for mf in meta_files:
        table=pq.read_table(mf);n=len(table)
        pq.write_table(pa.Table.from_pylist(records[offset:offset+n],schema=table.schema),mf);offset+=n
    stats=json.loads((a.output/'meta/stats.json').read_text())
    agg=aggregate_stats(image_stats)[video_key]
    stats[video_key]={k:v.tolist() for k,v in agg.items()}
    (a.output/'meta/stats.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    print(f'Processed {len(records)} episodes; masks match inference profile {a.task}')
if __name__=='__main__':main()
