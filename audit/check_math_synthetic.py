"""Independent synthetic-only numeric checks. No dataset arrays are opened."""
from pathlib import Path
import sys,json
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from model import TCN
from prepare import spike_counts,merge_intervals,covered,behavior_bin
from engine import metric,scaler,transform,blend_candidates
r=np.random.default_rng(191);n=TCN(seed=77,hidden=3,input_features=2)
x=r.normal(size=(4,7,2));y=r.normal(size=(4,2));loss,g=n.gradient(x,y)
maxerr=0;checked=0
for k in n.p:
 for ix in np.ndindex(n.p[k].shape):
  orig=n.p[k][ix];eps=1e-6
  n.p[k][ix]=orig+eps;pos=np.mean((n.forward(x)[0]-y)**2)
  n.p[k][ix]=orig-eps;neg=np.mean((n.forward(x)[0]-y)**2)
  n.p[k][ix]=orig;numeric=(pos-neg)/(2*eps)
  maxerr=max(maxerr,abs(numeric-g[k][ix]));checked+=1
assert maxerr<1e-7,maxerr
assert np.array_equal(spike_counts(np.array([0.,.05,.1,.15]),np.array([0.,.05,.1,.15])),[1,1,1])
ints=merge_intervals([[0,.05],[.05,.10],[.11,.20]])
assert np.array_equal(covered(ints,np.array([0,.05,.09,.11]),np.array([.1,.1,.12,.2])),[True,True,False,True])
t=np.arange(50)*.001;v=np.column_stack([t,-t]);assert np.allclose(behavior_bin(t,v,0,.05,.001),[.0245,-.0245])
assert behavior_bin(t[2:],v[2:],0,.05,.001) is None
assert behavior_bin(t[:-2],v[:-2],0,.05,.001) is None
keep=np.r_[0:20,24:50];assert behavior_bin(t[keep],v[keep],0,.05,.001) is None
v[3,0]=np.nan;assert behavior_bin(t,v,0,.05,.001) is None
pred=np.array([[1.,1.],[2.,4.],[3.,7.]]);actual=np.array([[1.,2.],[3.,4.],[5.,6.]])
m=metric(actual,pred);assert np.isclose(m['mse'],7/6);assert np.allclose(m['r2_per_axis'],[.375,.75])
weights=np.array(list(blend_candidates()));assert weights.shape==(15,3) and np.allclose(weights.sum(1),1) and np.all(weights>=0)
s=scaler({'counts':np.array([[0,1],[0,3],[0,2]]),'y':np.array([[1,2],[3,2],[5,2]])})
assert np.all(transform(np.array([[[4,99]]]),s)[...,0]==0)
assert s['ys'][1]==1
result={'pass':True,'checked_gradient_parameters':checked,'max_gradient_abs_error':maxerr,'checks':['TCN finite-difference gradient','half-open spike bins','union/gapped observation coverage','behavior bin edges, gaps, nonfinite rejection','native metrics','simplex grid','constant-channel transforms'],'dataset_arrays_opened':False}
(root/'audit/SYNTHETIC_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
