"""Full post-score verification. Run only after the original frozen final scoring is complete.
Independent algebra and NWB reconstruction; sklearn tree inference uses its saved estimator API.
"""
from pathlib import Path
import sys,json,hashlib,pickle,argparse
import numpy as np,h5py
P=Path(__file__).resolve().parents[1]
def j(p):return json.loads(Path(p).read_text())
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def close(a,b):assert np.allclose(a,b,rtol=1e-10,atol=1e-8),(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def metrics(y,p):
 err=(y-p)**2;m=err.mean();r=1-err.sum(0)/((y-y.mean(0))**2).sum(0)
 return {'mse':float(m),'rmse':float(m**.5),'r2_per_axis':r.tolist(),'mean_r2':float(r.mean()),'rows':len(y)}
def checkmetric(expected,y,p):
 for k,v in metrics(y,p).items():close(expected[k],v)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--post-score-confirmed',action='store_true');parser.add_argument('--results',default='scores');parser.add_argument('--raw',type=Path,help='Exact official NWB outside or inside the evidence package');parser.add_argument('--report',type=Path,default=Path('audit/POSTSCORE_REPLAY_AUDIT.json'),help='New report path; relative paths are relative to the package root; existing files refused');args=parser.parse_args()
 report_path=args.report if args.report.is_absolute() else P/args.report
 if report_path.exists():raise FileExistsError('Refusing to overwrite existing evidence: '+str(report_path))
 assert args.post_score_confirmed,'Explicit original-scoring confirmation required'
 out=P/args.results;assert (out/'RESULTS.json').exists(),'Original results absent; refusing target access'
 sys.path.insert(0,str(P))
 import study
 study.verify_models(P/'models')
 model=P/'models'
 meta=j(P/'data/metadata.json');data={role:load(P/'data'/f'{role}.npz') for role in ['fit','validation','evaluation']}
 for role in data:assert h(P/'data'/f'{role}.npz')==meta['splits'][role]['sha256']
 d=data['evaluation'];pred=load(out/'predictions.npz');res=j(out/'RESULTS.json');choices=j(model/'choices.json');s=load(model/'scaler.npz');plan=j(P/'PLAN.json')
 assert res['model_freeze_sha256']==h(model/'MODEL_FREEZE.json')
 for key in ['y','trial_id','bin_end','bin_index','source_trial_row']:assert np.array_equal(d[key],pred[key]),key
 # Independently recompute fitting-only statistics, with no pooling of validation or final data.
 counts=data['fit']['counts'];target=data['fit']['y'];sd=counts.std(0);ys=target.std(0)
 expected={'raw_mean':counts.mean(0),'raw_scale':np.where(sd==0,1,sd),'sqrt_mean':np.sqrt(counts).mean(0),'sqrt_scale':np.maximum(np.sqrt(counts).std(0),.25),'zero':sd==0,'ym':target.mean(0),'ys':np.where(ys==0,1,ys)}
 for key in expected:close(s[key],expected[key])
 # Check validation selection is first minimum in the complete predeclared grid.
 for name in ['ridge_current','ridge_history','trees']:
  c=choices[name];idx=int(np.argmin([z['validation_mse'] for z in c['grid']]));best=c['grid'][idx]
  close(c['validation_mse'],best['validation_mse'])
  for key,value in c['selected'].items():assert best[key]==value
  expected_n=18 if name.startswith('ridge') else 4;assert len(c['grid'])==expected_n
 for seed in plan['seeds']:
  c=choices['tcn_seed'+str(seed)];flat=[(v,i,ep+1) for i,z in enumerate(c['grid']) for ep,v in enumerate(z['validation_curve'])]
  best=min(flat,key=lambda z:z[0]);assert c['selected']['config']==best[1] and c['selected']['epoch']==best[2];close(c['validation_mse'],best[0]);assert len(flat)==160
 c=choices['blend'];best=min(c['grid'],key=lambda z:z['validation_mse']);close(c['weights'],best['weights']);close(c['validation_mse'],best['validation_mse']);assert len(c['grid'])==15
 # Independent prediction streams from saved parameters, without original model/engine/study imports.
 def transform(x,kind):
  if kind=='raw':z=(x-s['raw_mean'])/s['raw_scale']
  else:z=np.clip((np.sqrt(x)-s['sqrt_mean'])/s['sqrt_scale'],-5,5)
  z[...,s['zero']]=0;return z
 def infer(x):
  result={}
  for name,xx in [('ridge_current',x[:,-1:]),('ridge_history',x)]:
   z=transform(xx,choices[name]['selected']['kind']).reshape(len(x),-1);w=load(model/(name+'.npz'))['w'];result[name]=(z@w[:-1]+w[-1])*s['ys']+s['ym']
  z=transform(x,'robust')
  for seed in plan['seeds']:
   name='tcn_seed'+str(seed);p=load(model/(name+'.npz'));stage1=[]
   for t in range(5):stage1.append(np.maximum(np.einsum('bki,kih->bh',z[:,t:t+3,:],p['w1'])+p['b1'],0))
   a=np.stack(stage1,axis=1);b=np.maximum(np.einsum('bki,kih->bh',a[:,[0,2,4],:],p['w2'])+p['b2'],0)
   result[name]=(b@p['w3']+p['b3'])*s['ys']+s['ym']
  result['tcn_ensemble']=sum(result['tcn_seed'+str(seed)] for seed in plan['seeds'])/3
  with (model/'trees.pkl').open('rb') as f:trees=pickle.load(f)
  result['trees']=np.stack([t.predict(transform(x,'raw').reshape(len(x),-1)) for t in trees],axis=1)*s['ys']+s['ym']
  result['blend']=sum(w*result[k] for w,k in zip(choices['blend']['weights'],['ridge_history','tcn_ensemble','trees']))
  result['fit_mean']=np.broadcast_to(s['ym'],(len(x),2)).copy();return result
 streams=infer(d['x']);validation=infer(data['validation']['x']);vp=load(model/'validation.npz')
 for name,p in streams.items():close(p,pred[name]);checkmetric(res['metrics'][name],d['y'],p)
 for name,p in validation.items():
  if name!='fit_mean':close(p,vp[name]);close(metrics(data['validation']['y'],p)['mse'],choices[name]['validation_mse'])
 trialrows=np.unique(d['source_trial_row']);pertrial=j(out/'PER_TRIAL.json');assert len(pertrial)==len(trialrows)
 for i,row in enumerate(trialrows):
  mask=d['source_trial_row']==row;entry=pertrial[i];assert entry['source_trial_row']==row and entry['trial_id']==d['trial_id'][mask][0] and entry['rows']==mask.sum()
  for name,p in streams.items():close(entry['mse'][name],np.mean((d['y'][mask]-p[mask])**2))
 blocks=j(out/'CHRONOLOGICAL_BLOCKS.json');assert len(blocks)==10
 for i,rows in enumerate(np.array_split(trialrows,10)):
  b=blocks[i];mask=np.isin(d['source_trial_row'],rows);assert b['block']==i and b['first_trial']==rows[0] and b['last_trial']==rows[-1] and b['trials']==len(rows)
  for name,p in streams.items():checkmetric(b['metrics'][name],d['y'][mask],p[mask])
 primary=res['primary'];m=metrics(d['y'],streams['tcn_ensemble'])['mse'];ridge=metrics(d['y'],streams['ridge_history'])['mse'];tree=metrics(d['y'],streams['trees'])['mse'];assert primary['passed']==bool(m<ridge and m<tree);close(primary['mse_reduction_vs_history_ridge'],1-m/ridge);close(primary['mse_reduction_vs_trees'],1-m/tree)
 # Full final raw row rebuild, independent histogram counts and source slicing.
 raw=args.raw if args.raw is not None else P/'raw/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb';protocol=j(P/'SOURCE.json');assert h(raw)==protocol['source_sha256']
 checked=0
 with h5py.File(raw,'r') as f:
  tr=f['intervals/trials'];starts=tr['start_time'][:];stops=tr['stop_time'][:];ids=tr['id'][:];labels=tr['split'][:];moves=tr['move_onset_time'][:]
  official=np.array([i for i,l in enumerate(labels) if l in [b'train','train']]);official=official[np.lexsort((ids[official],starts[official]))];a=int(len(official)*.6);b=a+int(len(official)*.2)
  for role,nominal,prev in [('fit',official[:a],None),('validation',official[a:b],stops[official[a-1]]),('evaluation',official[b:],stops[official[b-1]])]:
   assert np.array_equal(nominal,meta['splits'][role]['source_trial_rows']);used=nominal if prev is None else nominal[starts[nominal]>=prev+2]
   assert np.array_equal(np.unique(data[role]['source_trial_row']),used)
  u=f['units'];uid=u['id'][:];cols=np.array([i for i,k in enumerate(uid) if str(int(k)).startswith('2')]);assert np.array_equal(uid[cols],d['unit_ids'])
  si=u['spike_times_index'][:];oi=u['obs_intervals_index'][:];spikes=[];obs=[]
  for col in cols:
   spikes.append(u['spike_times'][int(si[col-1]) if col else 0:int(si[col])]);obs.append(u['obs_intervals'][int(oi[col-1]) if col else 0:int(oi[col])])
  times=f['processing/behavior/hand_vel/timestamps'][:];values=f['processing/behavior/hand_vel/data'];dt=np.median(np.diff(times))
  for row in trialrows:
   indices=np.flatnonzero(d['source_trial_row']==row);edges=moves[row]+(-.25+np.arange(15)*.05)
   assert np.array_equal(d['bin_index'][indices],np.arange(6,14));assert edges[0]>=starts[row]-1e-9 and edges[-1]<=stops[row]+1e-9
   counts=np.zeros((14,len(cols)),int)
   for c,sp in enumerate(spikes):
    lo,hi=np.searchsorted(sp,[edges[0],edges[-1]],side='left');counts[:,c]=np.histogram(sp[lo:hi],bins=edges)[0]
    # Entire 700ms window coverage by union of intervals, independently walked.
    cursor=edges[0]
    for aa,bb in sorted(obs[c].tolist()):
     if bb<cursor-1e-9:continue
     if aa>cursor+1e-9:break
     cursor=max(cursor,bb)
     if cursor>=edges[-1]-1e-9:break
    assert cursor>=edges[-1]-1e-9
   for ix in indices:
    k=d['bin_index'][ix];assert np.array_equal(counts[k-6:k+1],d['x'][ix]);lo,hi=np.searchsorted(times,[edges[k],edges[k+1]],side='left');ts=times[lo:hi];assert len(ts) and ts[0]<=edges[k]+1.01*dt and ts[-1]>=edges[k+1]-1.01*dt and np.max(np.diff(ts))<=2*dt+1e-12
    vv=values[int(lo):int(hi)];assert np.isfinite(vv).all();target=(np.mean(vv,axis=0)*float(values.attrs.get('conversion',1))+float(values.attrs.get('offset',0)))*1000
    close(target,d['y'][ix]);assert ids[row]==d['trial_id'][ix];close(edges[k+1],d['bin_end'][ix]);checked+=1
 report={'pass':True,'evaluation_rows_raw_rebuilt':checked,'evaluation_trials':len(trialrows),'all_final_streams_independently_recomputed':list(streams),'all_validation_streams_recomputed':True,'fit_only_scaling_verified':True,'all_selection_minima_verified':True,'all_per_trial_and_ten_block_metrics_verified':True,'primary':primary,'source_sha256_verified':True,'limits':['Saved sklearn tree estimator inference reused; no independent tree-engine implementation.','Selection minima checked against complete saved validation grids; candidate models not refitted.','Raw reconstruction covers every evaluation row and six separately checked fitting rows, not all training rows.','One successful-trial source subject/session; not continuous or unseen-condition evaluation.']}
 report_path.parent.mkdir(parents=True,exist_ok=True)
 with report_path.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
