import numpy as np

class TCN:
 def __init__(self,seed=7,hidden=16,input_features=96):
  r=np.random.default_rng(seed); self.p={'w1':r.normal(0,np.sqrt(2/(3*input_features)),(3,input_features,hidden)), 'b1':np.zeros(hidden),'w2':r.normal(0,np.sqrt(2/(3*hidden)),(3,hidden,hidden)), 'b2':np.zeros(hidden),'w3':r.normal(0,.1,(hidden,2)),'b3':np.zeros(2)}
 def forward(self,x):
  p=self.p;z1=sum(x[:,k:k+5]@p['w1'][k] for k in range(3))+p['b1'];h1=np.maximum(z1,0)
  z2=sum(h1[:,2*k]@p['w2'][k] for k in range(3))+p['b2'];h2=np.maximum(z2,0)
  return h2@p['w3']+p['b3'],(x,z1,h1,z2,h2)
 def gradient(self,x,y):
  pred,c=self.forward(x);x,z1,h1,z2,h2=c;d=(pred-y)/len(y);p=self.p
  g={'w3':h2.T@d,'b3':d.sum(0)};d2=(d@p['w3'].T)*(z2>0)
  g['b2']=d2.sum(0);g['w2']=np.stack([h1[:,2*k].T@d2 for k in range(3)])
  dh=np.zeros_like(h1)
  for k in range(3):dh[:,2*k]+=d2@p['w2'][k].T
  dz=dh*(z1>0);g['b1']=dz.sum((0,1));g['w1']=np.stack([np.einsum('bti,bto->io',x[:,k:k+5],dz) for k in range(3)])
  return float(np.mean((pred-y)**2)),g

