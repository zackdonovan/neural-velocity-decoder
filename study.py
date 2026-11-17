"""Train source-specific decoders; freeze all choices before final scoring."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,hashlib,pickle,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import engine as e
from model import TCN
ROOT=Path(__file__).resolve().parent

def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify_code():
    seal=json.loads((ROOT/'PUBLIC_CODE_MANIFEST.json').read_text())
    for name,h in seal['sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Changed implementation/protocol: '+name)
def verify_models(folder):
    verify_code();f=json.loads((folder/'MODEL_FREEZE.json').read_text())
    expected=digest(ROOT/'PUBLIC_CODE_MANIFEST.json')
    if folder.resolve()==(ROOT/'models').resolve():
        expected=json.loads((ROOT/'PUBLIC_PACKAGE.json').read_text())['original_implementation_manifest_sha256']
    if f['implementation_seal_sha256']!=expected:raise ValueError('Wrong model manifest provenance')
    for name,h in f['input_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Fitting input changed: '+name)
    for name,h in f['model_sha256'].items():
        if digest(folder/name)!=h:raise ValueError('Model changed: '+name)
    return f
def current(d):return {**d,'x':d['x'][:,-1:]}
def fit(out):
    verify_code();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    tr=arrays(ROOT/'data/fit.npz');va=arrays(ROOT/'data/validation.npz')
    meta=json.loads((ROOT/'data/metadata.json').read_text())
    for role in ['fit','validation']:
        if digest(ROOT/'data'/(role+'.npz'))!=meta['splits'][role]['sha256']:raise ValueError('Prepared input changed')
    if not np.array_equal(tr['unit_ids'],va['unit_ids']):raise ValueError('Unit order mismatch')
    s=e.scaler(tr);np.savez_compressed(out/'scaler.npz',**s)
    choices={};pv={};started=time.time()
    for name,a,b in [('ridge_current',current(tr),current(va)),('ridge_history',tr,va)]:
        w,p,c=e.train_ridge(a,b,s);np.savez_compressed(out/(name+'.npz'),w=w);pv[name]=p;choices[name]=c
        print('FIT',name,'validation MSE',c['validation_mse'],flush=True)
    for seed in e.PLAN['seeds']:
        name='tcn_seed'+str(seed);start=time.time();p,pred,c=e.train_tcn(tr,va,s,seed=seed)
        np.savez_compressed(out/(name+'.npz'),**p);pv[name]=pred;choices[name]=c
        print('FIT',name,'validation MSE',c['validation_mse'],'seconds',time.time()-start,flush=True)
    pv['tcn_ensemble']=np.mean([pv['tcn_seed'+str(seed)] for seed in e.PLAN['seeds']],axis=0)
    choices['tcn_ensemble']=dict(seeds=e.PLAN['seeds'],weights=[1/3]*3,validation_mse=e.mse(va['y'],pv['tcn_ensemble']))
    nets,p,c=e.train_tree(tr,va,s);pv['trees']=p;choices['trees']=c
    with (out/'trees.pkl').open('wb') as f:pickle.dump(nets,f,protocol=5)
    best=float('inf');grid=[]
    for weights in e.blend_candidates():
        pred=sum(w*pv[k] for w,k in zip(weights,['ridge_history','tcn_ensemble','trees']))
        loss=e.mse(va['y'],pred);grid.append(dict(weights=weights.tolist(),validation_mse=loss))
        if loss<best:best=loss;bw=weights.copy();bp=pred.copy()
    pv['blend']=bp;choices['blend']=dict(weights=bw.tolist(),validation_mse=best,grid=grid)
    choices['selected_non_neural']=min(['ridge_current','ridge_history','trees'],key=lambda k:choices[k]['validation_mse'])
    choices['fit_rows']=len(tr['y']);choices['validation_rows']=len(va['y']);choices['input_units']=tr['x'].shape[-1]
    choices['seconds_elapsed']=time.time()-started
    dump(out/'choices.json',choices);np.savez_compressed(out/'validation.npz',y=va['y'],trial_id=va['trial_id'],bin_end=va['bin_end'],**pv)
    files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    dump(out/'MODEL_FREEZE.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),implementation_seal_sha256=digest(ROOT/'PUBLIC_CODE_MANIFEST.json'),
         input_sha256={name:digest(ROOT/name) for name in ['data/fit.npz','data/validation.npz','data/metadata.json','data/TRIAL_LEDGER.json']},model_sha256=files))
    print('ALL MODEL CHOICES FROZEN',flush=True)

def predict(d,folder):
    s=arrays(folder/'scaler.npz');c=json.loads((folder/'choices.json').read_text())
    if d['x'].shape[-1]!=c['input_units']:raise ValueError('Wrong neural dimension')
    result={}
    for name,dd in [('ridge_current',current(d)),('ridge_history',d)]:
        w=arrays(folder/(name+'.npz'))['w'];result[name]=e.native(e.design(e.transform(dd['x'],s,c[name]['selected']['kind']))@w,s)
    for seed in e.PLAN['seeds']:
        name='tcn_seed'+str(seed);net=TCN(hidden=c[name]['selected']['hidden'],input_features=c['input_units']);net.p=arrays(folder/(name+'.npz'))
        result[name]=e.native(net.forward(e.transform(d['x'],s))[0],s)
    result['tcn_ensemble']=np.mean([result['tcn_seed'+str(seed)] for seed in e.PLAN['seeds']],axis=0)
    with (folder/'trees.pkl').open('rb') as f:nets=pickle.load(f)
    x=e.transform(d['x'],s,'raw').reshape(len(d['x']),-1)
    result['trees']=e.native(np.column_stack([n.predict(x) for n in nets]),s)
    result['blend']=sum(w*result[k] for w,k in zip(c['blend']['weights'],['ridge_history','tcn_ensemble','trees']))
    result['fit_mean']=np.tile(s['ym'],(len(d['x']),1))
    return result

def score(folder,out):
    folder,out=Path(folder),Path(out);freeze=verify_models(folder)
    meta=json.loads((ROOT/'data/metadata.json').read_text())
    source=ROOT/'data/evaluation.npz'
    if digest(source)!=meta['splits']['evaluation']['sha256']:raise ValueError('Evaluation arrays changed')
    d=arrays(source);tr=arrays(ROOT/'data/fit.npz')
    if not np.array_equal(tr['unit_ids'],d['unit_ids']):raise ValueError('Unit identity mismatch')
    out.mkdir(parents=True,exist_ok=False);p=predict(d,folder)
    np.savez_compressed(out/'predictions.npz',y=d['y'],trial_id=d['trial_id'],bin_end=d['bin_end'],bin_index=d['bin_index'],source_trial_row=d['source_trial_row'],**p)
    metrics={k:e.metric(d['y'],v) for k,v in p.items()}
    trials=np.unique(d['source_trial_row']);pertrial=[]
    for trial in trials:
        ix=d['source_trial_row']==trial
        pertrial.append(dict(source_trial_row=int(trial),trial_id=int(d['trial_id'][ix][0]),rows=int(ix.sum()),mse={k:e.mse(d['y'][ix],v[ix]) for k,v in p.items()}))
    blocks=[]
    for j,group in enumerate(np.array_split(trials,10)):
        ix=np.isin(d['source_trial_row'],group)
        blocks.append(dict(block=j,first_trial=int(group[0]),last_trial=int(group[-1]),trials=len(group),metrics={k:e.metric(d['y'][ix],v[ix]) for k,v in p.items()}))
    choices=json.loads((folder/'choices.json').read_text());m=metrics['tcn_ensemble']['mse']
    primary=dict(criterion='Three-seed TCN ensemble MSE lower than both history ridge and trees on the fixed evaluation trials.',passed=m<metrics['ridge_history']['mse'] and m<metrics['trees']['mse'],
                 mse_reduction_vs_history_ridge=1-m/metrics['ridge_history']['mse'],mse_reduction_vs_trees=1-m/metrics['trees']['mse'],validation_selected_non_neural=choices['selected_non_neural'])
    dump(out/'RESULTS.json',dict(metrics=metrics,primary=primary,trial_count=len(trials),rows=len(d['y']),units='mm/s',
         model_freeze_sha256=digest(folder/'MODEL_FREEZE.json'),scored_utc=datetime.now(timezone.utc).isoformat(),scope='Fresh to this project, one new subject/task/session; chronological trials after new-source training, not cross-animal weight transfer or unseen conditions.'))
    dump(out/'PER_TRIAL.json',pertrial);dump(out/'CHRONOLOGICAL_BLOCKS.json',blocks)
    print(json.dumps(dict(metrics=metrics,primary=primary),indent=2),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['fit','score']);parser.add_argument('--output',required=True);parser.add_argument('--models',default=str(ROOT/'models'));args=parser.parse_args()
    fit(args.output) if args.mode=='fit' else score(args.models,args.output)
