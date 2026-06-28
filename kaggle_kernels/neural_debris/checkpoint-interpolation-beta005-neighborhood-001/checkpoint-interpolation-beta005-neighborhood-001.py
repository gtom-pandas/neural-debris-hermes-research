# checkpoint_interpolation_beta005_neighborhood_001
# No-submit: micro-map around validated interp20_30_beta0p05, near-triangular interpolation, and beta0p05 raw-output postprocess.
# No training, no architecture/loss change, no Kaggle submission.
import subprocess, sys, os
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'setuptools<81'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/facebookresearch/detectron2.git'])

import csv, hashlib, json, math, time, statistics
from pathlib import Path
from typing import Dict, List, Tuple
import cv2, numpy as np, torch

torch.backends.nnpack.enabled = False
torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS','2')))
torch.set_num_interop_threads(int(os.environ.get('TORCH_NUM_INTEROP_THREADS','1')))
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm

RUN_ID='checkpoint_interpolation_beta005_neighborhood_001'
OUT=Path('/kaggle/working/artifacts')/RUN_ID
CKPT_OUT=OUT/'checkpoints'; CSV_OUT=OUT/'csv'; AUDIT_OUT=OUT/'audits'; RAW_OUT=OUT/'raw'
for d in [OUT,CKPT_OUT,CSV_OUT,AUDIT_OUT,RAW_OUT]: d.mkdir(parents=True, exist_ok=True)
COMP=Path('/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models')
SAMPLE=COMP/'sample_submission.csv'; TEST_DIR=COMP/'test_set/test_set'
SHA15='c89566925a43791377ec2ce4cf86048c6e37dc2112442a013b23606320f3ebea'
SHA20='41cf8f23b38722efa3aad2f024d02d1e874a3b3d0d1e74ecfb57d58cc910e578'
SHA30='53301dc0a7db6a3707139a97ba5c4518d5c258f0bd3328ad2bcb28a77e6a79d8'
SHA_MAX20_CSV='401188594773a307a5fc8e790a5eacd667ee252b1f6ed1918d3b2768442351ab'
SHA_R02='5ee41b823fdd052fdcf237d98d58dca1ccec9edbf1a82184e9b64934c42dbd60'
BASE_CONFIG='COCO-Detection/retinanet_R_50_FPN_3x.yaml'
ANCHOR_ASPECT_RATIOS=[0.1,0.2,0.5,1.0,2.0,5.0,10.0]
ANCHOR_SIZES=[[16],[32],[64],[128],[256]]
IMG_W=IMG_H=1024
BASELINE={
 'MAX_ITER_20': {'detections_total':1866,'empty_images':791,'confidence_sum':774.230623,'score_median':0.381315,'public':259.7886},
 'r02_cluster_visible_no_power': {'detections_total':1815,'empty_images':791,'confidence_sum':760.941533,'score_median':0.387101,'public':259.2333},
 'interp20_30_beta0p05': {'detections_total':1831,'empty_images':806,'confidence_sum':750.414979,'score_median':0.376143,'public':257.6416},
 'MAX_ITER_15': {'detections_total':2176,'empty_images':717,'confidence_sum':1013.814393,'score_median':0.439227,'public':290.9630},
 'MAX_ITER_30': {'detections_total':1086,'empty_images':1159,'confidence_sum':371.1147,'score_median':0.304377,'public':None},
}
PHASE1=[0.02,0.03,0.04,0.06,0.07,0.08]
TRI=[(0.01,0.94,0.05),(0.02,0.93,0.05),(0.03,0.92,0.05)]
POST_SPECS=[
 {'name':'beta0p05_r02_style','threshold':0.20,'topk':3,'power':1.0,'cluster':True,'hypothesis':'apply r02-style top3/threshold to beta0p05 raw outputs'},
 {'name':'beta0p05_top4','threshold':0.20,'topk':4,'power':1.0,'cluster':False,'hypothesis':'add one extra beta0p05 raw detection slot per image'},
 {'name':'beta0p05_top5','threshold':0.20,'topk':5,'power':1.0,'cluster':False,'hypothesis':'add two extra beta0p05 raw detection slots per image'},
 {'name':'beta0p05_thr0p18_top3','threshold':0.18,'topk':3,'power':1.0,'cluster':True,'hypothesis':'recover slight recall below 0.20 with r02-style top3'},
 {'name':'beta0p05_thr0p22_top3','threshold':0.22,'topk':3,'power':1.0,'cluster':True,'hypothesis':'slightly stricter beta0p05 r02-style suppression'},
 {'name':'beta0p05_thr0p20_top3_power1p1','threshold':0.20,'topk':3,'power':1.1,'cluster':True,'hypothesis':'defensive confidence compression on beta0p05 boxes'},
]

def sha256_file(path: Path) -> str:
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for c in iter(lambda:f.read(1024*1024), b''): h.update(c)
 return h.hexdigest()

def find_by_sha(target_sha: str, suffix: str) -> Path:
 seen=[]
 for p in Path('/kaggle/input').rglob(suffix):
  try: s=sha256_file(p)
  except Exception: continue
  seen.append({'path':str(p),'sha256':s,'size':p.stat().st_size})
  if s==target_sha:
   (AUDIT_OUT/f'find_{target_sha[:8]}.json').write_text(json.dumps({'target':target_sha,'found':str(p),'seen_count':len(seen)},indent=2))
   return p
 (AUDIT_OUT/f'find_{target_sha[:8]}_failed.json').write_text(json.dumps({'target':target_sha,'seen':seen[:200]},indent=2))
 raise FileNotFoundError(target_sha)

def read_sample(path):
 rows=list(csv.DictReader(open(path,newline='',encoding='utf-8')))
 assert rows and list(rows[0].keys())==['id','image_id','prediction_string']
 return rows

def load_16bit_scaled(path: Path) -> np.ndarray:
 im=cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
 if im is None: raise FileNotFoundError(path)
 if im.dtype==np.uint16: im=im.astype(np.float32)/65535.0
 im=np.clip(im*255,0,255).astype(np.float32)
 if im.ndim==2: im=np.repeat(im[:,:,None],3,axis=2)
 return im

def parse_pred(s: str):
 vals=str(s or '').strip().split()
 if not vals: return []
 nums=[float(v) for v in vals]
 if len(nums)%5: return []
 return [tuple(nums[i:i+5]) for i in range(0,len(nums),5)]

def dets_to_str(dets):
 parts=[]
 for d in dets:
  s,x,y,w,h=d
  if w<=0 or h<=0: continue
  parts += [f'{float(s):.6f}', f'{float(x):.2f}', f'{float(y):.2f}', f'{float(w):.2f}', f'{float(h):.2f}']
 return ' '.join(parts) if parts else ' '

def load_submission_map(path):
 m={}
 for r in csv.DictReader(open(path,newline='',encoding='utf-8')):
  m[str(r['image_id'])]=parse_pred(r.get('prediction_string',''))
 return m

def box_iou(a,b):
 ax1,ay1,aw,ah=a[1],a[2],a[3],a[4]; bx1,by1,bw,bh=b[1],b[2],b[3],b[4]
 ax2,ay2=ax1+aw,ay1+ah; bx2,by2=bx1+bw,by1+bh
 ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
 inter=max(0,ix2-ix1)*max(0,iy2-iy1); union=max(0,aw)*max(0,ah)+max(0,bw)*max(0,bh)-inter
 return inter/union if union>0 else 0.0

def greedy_match(ref,cand,thr=0.5):
 pairs=[]; used_r=set(); used_c=set(); allp=[]
 for i,r in enumerate(ref):
  for j,c in enumerate(cand):
   iou=box_iou(r,c)
   if iou>=thr: allp.append((iou,i,j))
 allp.sort(reverse=True)
 for iou,i,j in allp:
  if i not in used_r and j not in used_c:
   used_r.add(i); used_c.add(j); pairs.append((i,j,iou))
 return pairs,used_r,used_c

def spatial_compare(ref_map,cand_map,sample_rows,thr=0.5):
 rt=ct=matched=0; ious=[]; ratios=[]; dropped={}; new={}
 for r in sample_rows:
  img=str(r['image_id']); rd=ref_map.get(img,[]); cd=cand_map.get(img,[])
  rt+=len(rd); ct+=len(cd); pairs,ur,uc=greedy_match(rd,cd,thr); matched+=len(pairs)
  for i,j,iou in pairs:
   ious.append(iou)
   if rd[i][0]>0: ratios.append(cd[j][0]/rd[i][0])
  if len(rd)-len(ur): dropped[img]=len(rd)-len(ur)
  if len(cd)-len(uc): new[img]=len(cd)-len(uc)
 q=lambda vals,p: float(np.quantile(np.asarray(vals,dtype=np.float64),p)) if vals else None
 return {'ref_detections':rt,'candidate_detections':ct,'matched':matched,'recall_vs_ref':matched/rt if rt else 1.0,'precision_vs_ref':matched/ct if ct else 1.0,'dropped_ref_detections':rt-matched,'new_candidate_detections':ct-matched,'images_with_dropped_ref':len(dropped),'images_with_new_candidate':len(new),'matched_iou_median':q(ious,0.5),'matched_score_ratio_median':q(ratios,0.5),'matched_score_ratio_p10':q(ratios,0.1),'matched_score_ratio_p90':q(ratios,0.9)}

def pct(vals,q): return float(np.quantile(np.asarray(vals,dtype=np.float64),q)) if vals else None

def summarize_map(sub_map,sample_rows):
 scores=[]; tops=[]; counts=[]; empty=0; invalid=0; nonfinite=0
 for r in sample_rows:
  arr=sub_map.get(str(r['image_id']),[]); counts.append(len(arr)); empty += 1 if not arr else 0; img_scores=[]
  for d in arr:
   if len(d)!=5: invalid+=1; continue
   if not all(math.isfinite(float(x)) for x in d): nonfinite+=1
   scores.append(d[0]); img_scores.append(d[0])
  if img_scores: tops.append(max(img_scores))
 return {'rows':len(sample_rows),'detections_total':len(scores),'detections_per_image':len(scores)/len(sample_rows),'empty_images':empty,'invalid_prediction_groups':invalid,'nonfinite_values':nonfinite,'confidence_sum':float(sum(scores)),'score_mean':float(statistics.mean(scores)) if scores else 0.0,'score_median':float(statistics.median(scores)) if scores else 0.0,'score_p10':pct(scores,.1),'score_p90':pct(scores,.9),'top_score_median':float(statistics.median(tops)) if tops else 0.0,'per_image_count_p95':pct(counts,.95)}

def cpsr(stats, spatial20):
 det=stats['detections_total']; empty=stats['empty_images']; conf=stats['confidence_sum']; med=stats['score_median']; rec=spatial20['recall_vs_ref']; prec=spatial20['precision_vs_ref']
 hard=[]
 if det<=100 or empty>=1800 or conf<50 or rec<0.05: hard.append('collapse')
 if det<1600 or empty>950 or conf<620 or rec<0.85: hard.append('sur_suppression')
 if det>2100 or empty<730 or conf>950 or med>0.43 or prec<0.85: hard.append('inflation_or_precision_loss')
 suppression=max(0,(1750-det)/650)*35 + max(0,(empty-830)/400)*35 + max(0,(700-conf)/350)*25 + max(0,(0.97-rec)/0.40)*45
 inflation=max(0,(det-1950)/350)*35 + max(0,(760-empty)/120)*25 + max(0,(conf-825)/250)*35 + max(0,(med-0.42)/0.10)*20
 spatial=max(0,(0.97-rec)/0.12)*50 + max(0,(0.94-prec)/0.12)*50
 risk=min(100,0.45*min(100,suppression)+0.25*min(100,inflation)+0.30*min(100,spatial))
 if 1750<=det<=1950 and 760<=empty<=830 and 700<=conf<=825 and 0.35<=med<=0.42 and rec>=0.97 and prec>=0.94: zone='green'
 elif not hard and risk<=25: zone='orange_moderate'
 elif hard: zone='red_hard_reject'
 else: zone='red_high_risk'
 return {'cpsr_risk_0_100':risk,'zone':zone,'hard_reject_flags':hard,'suppression_component_raw':suppression,'inflation_component_raw':inflation,'spatial_component_raw':spatial}

def make_cfg(checkpoint, thresh):
 cfg=get_cfg(); cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG)); cfg.MODEL.WEIGHTS=str(checkpoint); cfg.MODEL.DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
 cfg.MODEL.RETINANET.NUM_CLASSES=1; cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS=[ANCHOR_ASPECT_RATIOS]; cfg.MODEL.ANCHOR_GENERATOR.SIZES=ANCHOR_SIZES; cfg.MODEL.RETINANET.SCORE_THRESH_TEST=float(thresh); cfg.DATASETS.TRAIN=(); cfg.DATASETS.TEST=()
 return cfg

def pred_from_instances(instances):
 out=instances.to('cpu'); boxes=out.pred_boxes.tensor.numpy(); scores=out.scores.numpy(); dets=[]
 for (x1,y1,x2,y2),s in zip(boxes,scores):
  x1=float(np.clip(x1,0,IMG_W)); y1=float(np.clip(y1,0,IMG_H)); x2=float(np.clip(x2,0,IMG_W)); y2=float(np.clip(y2,0,IMG_H)); w=max(0.0,x2-x1); h=max(0.0,y2-y1)
  if w>0 and h>0: dets.append((float(s),x1,y1,w,h))
 return dets

def interpolate_components(name, components):
 loaded=[]
 for coef,path in components:
  ck=torch.load(path,map_location='cpu'); loaded.append((float(coef),ck))
 keys=set(loaded[0][1]['model'].keys())
 for _,ck in loaded[1:]:
  if set(ck['model'].keys())!=keys: raise ValueError(f'key mismatch {name}')
 out={'model':{}}; float_tensors=0; copied=0
 for k in keys:
  tensors=[ck['model'][k] for _,ck in loaded]
  if all(torch.is_tensor(t) and torch.is_floating_point(t) for t in tensors):
   v=sum(coef*t.to(torch.float32) for coef,t in zip([c for c,_ in loaded],tensors)); out['model'][k]=v.to(dtype=tensors[0].dtype); float_tensors+=1
  else:
   out['model'][k]=tensors[-1]; copied+=1
 for meta in ['__author__','matching_heuristics']:
  if meta in loaded[-1][1]: out[meta]=loaded[-1][1][meta]
 p=CKPT_OUT/f'{name}.pth'; torch.save(out,p)
 audit={'variant':name,'components':[{'coef':c,'path':str(path),'sha256':sha256_file(path)} for c,path in components],'coef_sum':sum(c for c,_ in components),'output':str(p),'sha256':sha256_file(p),'float_tensors_interpolated':float_tensors,'nonfloat_copied_from_last':copied}
 (AUDIT_OUT/f'{name}_checkpoint_audit.json').write_text(json.dumps(audit,indent=2))
 return p,audit

def infer_checkpoint(name,checkpoint,sample_rows,thresh=0.2,save_raw=False):
 cfg=make_cfg(checkpoint,thresh); predictor=DefaultPredictor(cfg); csv_path=CSV_OUT/f'{name}.csv'; sub_map={}; raw_map={}; t0=time.perf_counter()
 with csv_path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['id','image_id','prediction_string']); w.writeheader()
  for idx,r in enumerate(tqdm(sample_rows,desc=f'infer {name}')):
   img=str(r['image_id']); im=load_16bit_scaled(TEST_DIR/f'{img}.png'); dets=pred_from_instances(predictor(im)['instances']); raw_map[img]=dets; ps=dets_to_str(dets); w.writerow({'id':r['id'],'image_id':img,'prediction_string':ps}); sub_map[img]=dets
   if idx%50==0: f.flush()
 if save_raw:
  raw_csv=RAW_OUT/f'{name}_raw_detections.csv'
  with raw_csv.open('w',newline='',encoding='utf-8') as f:
   w=csv.writer(f); w.writerow(['image_id','score','x','y','w','h'])
   for img,dets in raw_map.items():
    for d in dets: w.writerow([img,*d])
 else: raw_csv=None
 audit={'variant':name,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256_file(checkpoint),'threshold':thresh,'csv':str(csv_path),'csv_sha256':sha256_file(csv_path),'raw_csv':str(raw_csv) if raw_csv else None,'raw_sha256':sha256_file(raw_csv) if raw_csv else None,'runtime_sec':time.perf_counter()-t0,'submitted':False,'auto_submit':False,'submission_authorized':False}
 (AUDIT_OUT/f'{name}_inference_audit.json').write_text(json.dumps(audit,indent=2))
 return csv_path,sub_map,raw_map,audit

def cluster_dets(dets,iou_thr=0.6):
 # Greedy cluster/NMS: keep the highest-score representative per overlapping cluster.
 out=[]
 for d in sorted(dets,key=lambda x:x[0],reverse=True):
  if all(box_iou(d,o)<iou_thr for o in out): out.append(d)
 return out

def apply_postprocess(spec,raw_map,sample_rows):
 sub={}; csv_path=CSV_OUT/f"{spec['name']}.csv"
 with csv_path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['id','image_id','prediction_string']); w.writeheader()
  for r in sample_rows:
   img=str(r['image_id']); dets=[tuple(d) for d in raw_map.get(img,[]) if d[0]>=spec['threshold']]
   if spec.get('cluster'): dets=cluster_dets(dets)
   if spec.get('power',1.0)!=1.0: dets=[(float(d[0])**float(spec['power']),d[1],d[2],d[3],d[4]) for d in dets]
   dets=sorted(dets,key=lambda x:x[0],reverse=True)[:int(spec['topk'])]
   sub[img]=dets; w.writerow({'id':r['id'],'image_id':img,'prediction_string':dets_to_str(dets)})
 audit={'variant':spec['name'],'spec':spec,'csv':str(csv_path),'csv_sha256':sha256_file(csv_path),'submitted':False,'source':'beta0p05 raw threshold 0.02 outputs'}
 (AUDIT_OUT/f"{spec['name']}_postprocess_audit.json").write_text(json.dumps(audit,indent=2))
 return csv_path,sub,audit

def proximity(stats,sp20,ref):
 return abs(stats['detections_total']-ref['detections_total'])/200 + abs(stats['empty_images']-ref['empty_images'])/100 + abs(stats['confidence_sum']-ref['confidence_sum'])/150 + abs(stats['score_median']-ref['score_median'])/0.06 + abs(1-sp20['recall_vs_ref'])/0.05 + abs(1-sp20['precision_vs_ref'])/0.06

def add_row(rows,full,name,family,formula,hypothesis,csv_path,cmap,ckpt_audit,infer_audit,sample_rows,max20_map,r02_map,beta_ref_map=None):
 stats=summarize_map(cmap,sample_rows); sp20=spatial_compare(max20_map,cmap,sample_rows); spr02=spatial_compare(r02_map,cmap,sample_rows); c=cpsr(stats,sp20)
 spb=spatial_compare(beta_ref_map,cmap,sample_rows) if beta_ref_map else None
 row={'name':name,'family':family,'formula':formula,'hypothesis':hypothesis,**stats,**c,'recall_vs_max20':sp20['recall_vs_ref'],'precision_vs_max20':sp20['precision_vs_ref'],'dropped_max20':sp20['dropped_ref_detections'],'new_vs_max20':sp20['new_candidate_detections'],'recall_vs_r02':spr02['recall_vs_ref'],'precision_vs_r02':spr02['precision_vs_ref'],'distance_to_beta0p05':proximity(stats,sp20,BASELINE['interp20_30_beta0p05']),'distance_to_max20':proximity(stats,sp20,BASELINE['MAX_ITER_20']),'distance_to_r02':proximity(stats,sp20,BASELINE['r02_cluster_visible_no_power']),'delta_det_vs_beta0p05':stats['detections_total']-BASELINE['interp20_30_beta0p05']['detections_total'],'delta_empty_vs_beta0p05':stats['empty_images']-BASELINE['interp20_30_beta0p05']['empty_images'],'delta_conf_vs_beta0p05':stats['confidence_sum']-BASELINE['interp20_30_beta0p05']['confidence_sum'],'csv':str(csv_path),'csv_sha256':sha256_file(csv_path),'submitted':False}
 if spb:
  row.update({'recall_vs_beta0p05':spb['recall_vs_ref'],'precision_vs_beta0p05':spb['precision_vs_ref'],'dropped_beta0p05':spb['dropped_ref_detections'],'new_vs_beta0p05':spb['new_candidate_detections']})
 rows.append(row); full[name]={'summary_row':row,'stats':stats,'spatial_vs_max20':sp20,'spatial_vs_r02':spr02,'spatial_vs_beta0p05':spb,'cpsr':c,'checkpoint_audit':ckpt_audit,'generation_audit':infer_audit}
 (AUDIT_OUT/f'{name}_full_audit.json').write_text(json.dumps(full[name],indent=2))
 return row

def main():
 t0=time.perf_counter(); sample_rows=read_sample(SAMPLE); assert len(sample_rows)==2000
 ckpt15=find_by_sha(SHA15,'model_final.pth'); ckpt20=find_by_sha(SHA20,'model_final.pth'); ckpt30=find_by_sha(SHA30,'model_final.pth')
 max20_csv=find_by_sha(SHA_MAX20_CSV,'*.csv'); r02_csv=find_by_sha(SHA_R02,'*.csv')
 max20_map=load_submission_map(max20_csv); r02_map=load_submission_map(r02_csv)
 rows=[]; full={}; ckpts={'15':ckpt15,'20':ckpt20,'30':ckpt30}; beta_ref_map=None
 manifest={'run_id':RUN_ID,'mode':'no_submit_micro_map_beta005','submitted':False,'training':False,'architecture_change':False,'inputs':{'ckpt15':str(ckpt15),'ckpt20':str(ckpt20),'ckpt30':str(ckpt30),'max20_csv':str(max20_csv),'r02_csv':str(r02_csv)},'variants':{}}
 # Reference beta0p05 regenerated and raw-threshold inference for phase 3.
 beta_ckpt,beta_ca=interpolate_components('interp20_30_beta0p05_reference',[(0.95,ckpt20),(0.05,ckpt30)])
 beta_csv,beta_map,_,beta_ia=infer_checkpoint('interp20_30_beta0p05_reference',beta_ckpt,sample_rows,0.2,False)
 beta_ref_map=beta_map
 add_row(rows,full,'interp20_30_beta0p05_reference','reference_regen','0.95*W20+0.05*W30','regenerated active reference for distance/raw consistency',beta_csv,beta_map,beta_ca,beta_ia,sample_rows,max20_map,r02_map,None)
 _,_,beta_raw_map,beta_raw_ia=infer_checkpoint('interp20_30_beta0p05_raw0p02',beta_ckpt,sample_rows,0.02,True)
 # Phase 1 micro beta.
 for b in PHASE1:
  name=f"interp20_30_beta{str(b).replace('0.','0p')}"; ca_ckpt,ca=interpolate_components(name,[(1-b,ckpt20),(b,ckpt30)]); csvp,cmap,_,ia=infer_checkpoint(name,ca_ckpt,sample_rows,0.2,False)
  add_row(rows,full,name,'phase1_micro_beta',f'{1-b:.2f}*W20+{b:.2f}*W30',f'micro-map beta={b:.2f} around validated beta0p05',csvp,cmap,ca,ia,sample_rows,max20_map,r02_map,beta_ref_map); torch.cuda.empty_cache()
 # Phase 2 triangular.
 for a,b,c in TRI:
  name=f"tri_a{str(a).replace('0.','0p')}_b{str(b).replace('0.','0p')}_c{str(c).replace('0.','0p')}"; ck,ca=interpolate_components(name,[(a,ckpt15),(b,ckpt20),(c,ckpt30)]); csvp,cmap,_,ia=infer_checkpoint(name,ck,sample_rows,0.2,False)
  add_row(rows,full,name,'phase2_triangular',f'{a:.2f}*W15+{b:.2f}*W20+{c:.2f}*W30','slight MAX15 recall reinjection while keeping c=0.05 suppressive component',csvp,cmap,ca,ia,sample_rows,max20_map,r02_map,beta_ref_map); torch.cuda.empty_cache()
 # Phase 3 raw beta0p05 postprocess.
 for spec in POST_SPECS:
  csvp,cmap,pa=apply_postprocess(spec,beta_raw_map,sample_rows)
  add_row(rows,full,spec['name'],'phase3_beta0p05_postprocess',json.dumps({k:spec[k] for k in ['threshold','topk','power','cluster']}),spec['hypothesis'],csvp,cmap,None,pa,sample_rows,max20_map,r02_map,beta_ref_map)
 # Ranking: prefer green, proximity to beta, information distance, and plausible surpass score heuristic.
 for r in rows:
  risk_pen=0 if r['zone']=='green' else (25 if r['zone']=='orange_moderate' else 100)
  behavior=abs(r.get('delta_det_vs_beta0p05',0))/80 + abs(r.get('delta_conf_vs_beta0p05',0))/60 + abs(r.get('delta_empty_vs_beta0p05',0))/50
  plausibility = -risk_pen - r['distance_to_beta0p05'] + min(2.0, behavior)
  r['leaderboard_candidate_score_heuristic']=plausibility
  r['risk_estimate']='low' if r['zone']=='green' and r['distance_to_beta0p05']<1.0 else ('medium' if r['zone'] in ['green','orange_moderate'] else 'high')
  r['submission_recommendation']='candidate_review' if r['zone']=='green' and r['name']!='interp20_30_beta0p05_reference' else 'no_submit_now'
 ranked=sorted(rows,key=lambda x:(x['zone']!='green', x['distance_to_beta0p05']))
 top10=ranked[:10]
 top2=[]
 for r in sorted([x for x in rows if x['zone']=='green' and x['name']!='interp20_30_beta0p05_reference'], key=lambda x:x['leaderboard_candidate_score_heuristic'], reverse=True):
  if not top2: top2.append(r)
  elif abs(r['delta_det_vs_beta0p05']-top2[0]['delta_det_vs_beta0p05'])>10 or abs(r['delta_conf_vs_beta0p05']-top2[0]['delta_conf_vs_beta0p05'])>10 or r['family']!=top2[0]['family']:
   top2.append(r); break
 fields=[]
 for rr in rows:
  for k in rr.keys():
   if k not in fields: fields.append(k)
 for fname, data in [('candidate_summary_all.csv', rows),('ranking_by_proximity_beta0p05.csv', ranked),('top10_candidates.csv', top10),('top2_submit_review.csv', top2)]:
  with (OUT/fname).open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(data)
 report={'run_id':RUN_ID,'submitted':False,'runtime_sec':time.perf_counter()-t0,'baseline':BASELINE,'rows':rows,'ranking_by_proximity_beta0p05':ranked,'top10':top10,'top2_submit_review':top2,'answer_options':{'A':'local optimum around 0.03-0.08','B':'general signal from light suppression plus postprocess'},'interpretation':'computed by post-run audit; no leaderboard submission performed'}
 (OUT/'final_audit_report.json').write_text(json.dumps(report,indent=2))
 manifest['sha256_outputs']={}
 for p in OUT.rglob('*'):
  if p.is_file():
   try: manifest['sha256_outputs'][str(p.relative_to(OUT))]=sha256_file(p)
   except Exception: pass
 (OUT/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({'run_id':RUN_ID,'submitted':False,'out':str(OUT),'top2':[r['name'] for r in top2],'runtime_sec':report['runtime_sec']},indent=2),flush=True)
if __name__=='__main__': main()
