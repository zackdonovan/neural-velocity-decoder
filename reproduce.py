"""Retrain and replay the fixed study from packaged arrays, never new evidence."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,json,subprocess,sys,pickle
import numpy as np
ROOT=Path(__file__).resolve().parent

def compare_npz(a,b):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        if set(x.files)!=set(y.files):raise AssertionError('Different array names: '+str(a))
        for k in x.files:
            if not np.array_equal(x[k],y[k]):raise AssertionError('Numeric array differs: '+str(a)+' '+k)
        return len(x.files)
def compare_trees(a,b):
    with a.open('rb') as f:x=pickle.load(f)
    with b.open('rb') as f:y=pickle.load(f)
    arrays=0
    for m,n in zip(x,y):
        assert m.n_features_in_==n.n_features_in_
        assert np.array_equal(m._baseline_prediction,n._baseline_prediction);arrays+=1
        assert len(m._predictors)==len(n._predictors)
        for s,t in zip(m._predictors,n._predictors):
            assert len(s)==len(t)
            for u,v in zip(s,t):
                assert u.nodes.dtype==v.nodes.dtype and u.nodes.shape==v.nodes.shape and u.nodes.tobytes()==v.nodes.tobytes();arrays+=1
        for s,t in zip(m._bin_mapper.bin_thresholds_,n._bin_mapper.bin_thresholds_):
            assert np.array_equal(s,t);arrays+=1
    return arrays
def main():
    a=argparse.ArgumentParser();a.add_argument('output',type=Path);args=a.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    subprocess.run([sys.executable,str(ROOT/'study.py'),'fit','--output',str(out/'models')],check=True)
    subprocess.run([sys.executable,str(ROOT/'study.py'),'score','--models',str(out/'models'),'--output',str(out/'scores')],check=True)
    n=0
    for p in (ROOT/'models').glob('*.npz'):n+=compare_npz(p,out/'models'/p.name)
    predictions=compare_npz(ROOT/'scores/predictions.npz',out/'scores/predictions.npz')
    trees=compare_trees(ROOT/'models/trees.pkl',out/'models/trees.pkl')
    old=json.loads((ROOT/'models/choices.json').read_text());new=json.loads((out/'models/choices.json').read_text())
    old.pop('seconds_elapsed');new.pop('seconds_elapsed');assert old==new
    old=json.loads((ROOT/'scores/RESULTS.json').read_text());new=json.loads((out/'scores/RESULTS.json').read_text())
    for k in ['metrics','primary','trial_count','rows','units','scope']:assert old[k]==new[k]
    for name in ['PER_TRIAL.json','CHRONOLOGICAL_BLOCKS.json']:
        assert json.loads((ROOT/'scores'/name).read_text())==json.loads((out/'scores'/name).read_text())
    report=dict(status='PASS',parameter_scaler_validation_arrays=n,prediction_target_index_arrays=predictions,tree_numeric_arrays=trees,
                scope='Exact numerical full retraining and inference reproduction of the same exposed evidence. Runtime/file timestamps ignored; scientific choices, curves and metrics identical.')
    (out/'REPRODUCTION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
