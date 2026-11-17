"""Model fitting algebra inherited from Indy benchmark, with variable input count."""
from pathlib import Path
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from model import TCN
PLAN=json.loads(Path(__file__).with_name('PLAN.json').read_text())
def scaler(tr):
 c,y=tr['counts'],tr['y'];sd,ys=c.std(0),y.std(0)
 return dict(raw_mean=c.mean(0),raw_scale=np.where(sd==0,1,sd),sqrt_mean=np.sqrt(c).mean(0),sqrt_scale=np.maximum(np.sqrt(c).std(0),.25),zero=sd==0,ym=y.mean(0),ys=np.where(ys==0,1,ys))
def transform(x,s,kind='robust'):
 z=(x-s['raw_mean'])/s['raw_scale'] if kind=='raw' else np.clip((np.sqrt(x)-s['sqrt_mean'])/s['sqrt_scale'],-5,5)
 z[...,s['zero']]=0
 return z

def native(p,s):return p*s['ys']+s['ym']
def mse(y,p):
 if y.shape!=p.shape or not len(y) or not np.isfinite(p).all():raise ValueError('Invalid prediction')
 return float(np.mean((y-p)**2))
def metric(y,p):
 loss=mse(y,p);den=np.sum((y-y.mean(0))**2,axis=0)
 if (den<=0).any():raise ValueError('Degenerate outcome')
 r=1-np.sum((y-p)**2,axis=0)/den
 return dict(mse=loss,rmse=float(np.sqrt(loss)),r2_per_axis=r.tolist(),mean_r2=float(r.mean()),rows=len(y))
def design(x):return np.column_stack((x.reshape(len(x),-1),np.ones(len(x))))
def train_ridge(tr,va,s):
 best=float('inf');grid=[]
 target=(tr['y']-s['ym'])/s['ys']
 for kind in PLAN['ridge_features']:
  a,b=design(transform(tr['x'],s,kind)),design(transform(va['x'],s,kind));gram,rhs=a.T@a,a.T@target
  for lam in PLAN['ridge_lambdas']:
   reg=np.eye(a.shape[1])*lam;reg[-1,-1]=0;w=np.linalg.solve(gram+reg,rhs)
   pred=native(b@w,s);loss=mse(va['y'],pred);grid.append(dict(kind=kind,lam=lam,validation_mse=loss))
   if loss<best:best=loss;bw=w.copy();bp=pred.copy();choice=dict(kind=kind,lam=lam)
 return bw,bp,dict(selected=choice,validation_mse=best,grid=grid)
def train_tcn(tr,va,s,seed=7):
 opt=PLAN['tcn_optimizer'];x,xv=transform(tr['x'],s),transform(va['x'],s);y=(tr['y']-s['ym'])/s['ys']
 best=float('inf');grid=[]
 for ci,c in enumerate(PLAN['tcn_configs']):
  n=TCN(seed=seed,hidden=c['hidden'],input_features=tr['x'].shape[-1]);mom={k:np.zeros_like(v) for k,v in n.p.items()};var={k:v.copy() for k,v in mom.items()}
  rng=np.random.default_rng(seed);step=0;curve=[]
  for epoch in range(1,opt['epochs']+1):
   order=rng.permutation(len(y))
   for start in range(0,len(y),opt['batch']):
    ix=order[start:start+opt['batch']];_,g=n.gradient(x[ix],y[ix])
    for k in g:
     if k.startswith('w'):g[k]+=c['weight_penalty']*n.p[k]
    norm=np.sqrt(sum(np.sum(v*v) for v in g.values()));clip=min(1,opt['gradient_clip']/(norm+1e-12));step+=1
    for k in n.p:
     grad=g[k]*clip;mom[k]=.9*mom[k]+.1*grad;var[k]=.999*var[k]+.001*grad*grad
     n.p[k]-=opt['learning_rate']*(mom[k]/(1-.9**step))/(np.sqrt(var[k]/(1-.999**step))+1e-8)
   pred=native(n.forward(xv)[0],s);loss=mse(va['y'],pred);curve.append(loss)
   if loss<best:best=loss;bp=pred.copy();params={k:v.copy() for k,v in n.p.items()};choice=dict(config=ci,epoch=epoch,**c)
  grid.append(dict(config=ci,**c,validation_curve=curve))
 return params,bp,dict(selected=choice,validation_mse=best,grid=grid)
def train_tree(tr,va,s):
 x=transform(tr['x'],s,'raw').reshape(len(tr['y']),-1);xv=transform(va['x'],s,'raw').reshape(len(va['y']),-1)
 y=(tr['y']-s['ym'])/s['ys'];best=float('inf');grid=[]
 for c in PLAN['tree_configs']:
  nets=[HistGradientBoostingRegressor(**c,**PLAN['tree_options']).fit(x,y[:,j]) for j in range(2)]
  pred=native(np.column_stack([n.predict(xv) for n in nets]),s);loss=mse(va['y'],pred);grid.append(dict(**c,validation_mse=loss))
  if loss<best:best=loss;bn=nets;bp=pred.copy();choice=c.copy()
 return bn,bp,dict(selected=choice,validation_mse=best,grid=grid)
def blend_candidates():
 for a in range(5):
  for b in range(5-a):yield np.array([a,b,4-a-b],float)/4
