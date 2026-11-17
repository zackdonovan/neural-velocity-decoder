from pathlib import Path
import tempfile,sys,json,pickle,hashlib
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import study
from model import TCN
from sklearn.dummy import DummyRegressor
from engine import scaler
r=np.random.default_rng(8);x=r.poisson(1,(5,7,2));s=scaler({'counts':x.reshape(-1,2),'y':r.normal(size=(5,2))})
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp);np.savez(p/'scaler.npz',**s)
 c={'input_units':2,'blend':{'weights':[.25,.5,.25]}}
 for name,size in [('ridge_current',3),('ridge_history',15)]:
  c[name]={'selected':{'kind':'raw'}};np.savez(p/(name+'.npz'),w=r.normal(size=(size,2)))
 for seed in [7,23,51]:
  name='tcn_seed'+str(seed);c[name]={'selected':{'hidden':3}};np.savez(p/(name+'.npz'),**TCN(seed=seed,hidden=3,input_features=2).p)
 (p/'choices.json').write_text(json.dumps(c))
 trees=[DummyRegressor(strategy='constant',constant=i).fit(np.zeros((3,14)),np.zeros(3)) for i in [0,1]]
 with (p/'trees.pkl').open('wb') as f:pickle.dump(trees,f)
 a=study.predict({'x':x},p);b=study.predict({'x':x,'y':np.full((5,2),999999.)},p)
 assert a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)
 assert np.allclose(a['tcn_ensemble'],sum(a['tcn_seed'+str(seed)] for seed in [7,23,51])/3)
 assert np.allclose(a['blend'],.25*a['ridge_history']+.5*a['tcn_ensemble']+.25*a['trees'])
 try:study.predict({'x':np.zeros((5,7,3))},p)
 except ValueError:pass
 else:raise AssertionError('Wrong neural dimension accepted')
 old=study.ROOT;study.ROOT=p
 (p/'fixture.txt').write_text('original');h=study.digest(p/'fixture.txt')
 (p/'PUBLIC_CODE_MANIFEST.json').write_text(json.dumps({'sha256':{'fixture.txt':h}}));study.verify_code()
 (p/'fixture.txt').write_text('tampered')
 try:study.verify_code()
 except ValueError:pass
 else:raise AssertionError('Tampered implementation accepted')
 study.ROOT=old
result={'pass':True,'checks':['label-free prediction','target-value invariance','equal three-seed ensemble','secondary fixed blend','wrong feature count rejection','implementation tamper rejection'],'real_data_or_models_opened':False}
(root/'audit/WRAPPER_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
