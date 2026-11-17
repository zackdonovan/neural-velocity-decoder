"""Independently rebuild a few fitting rows only; no validation/evaluation arrays."""
from pathlib import Path
import json,numpy as np,h5py
root=Path(__file__).resolve().parents[1]
a=np.load(root/'data/fit.npz');checked=0;max_yerr=0
with h5py.File(root/'raw/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb','r') as f:
 u=f['units'];allids=u['id'][:];inds=u['spike_times_index'][:];tr=f['intervals/trials'];t=f['processing/behavior/hand_vel/timestamps'][:];v=f['processing/behavior/hand_vel/data']
 for row in [0,7,8,1000,len(a['x'])-8,len(a['x'])-1]:
  source=int(a['source_trial_row'][row]);assert tr['split'][source] in [b'train','train']
  onset=float(tr['move_onset_time'][source]);k=int(a['bin_index'][row]);edges=onset-.25+np.arange(k-6,k+2)*.05
  expected=np.zeros((7,len(a['unit_ids'])),int)
  for j,uid in enumerate(a['unit_ids']):
   i=np.flatnonzero(allids==uid)[0];sp=u['spike_times'][int(inds[i-1]) if i else 0:int(inds[i])]
   for b in range(7):expected[b,j]=np.sum((sp>=edges[b])&(sp<edges[b+1]))
  # Float operation grouping differs from parser; no recorded spikes at the tiny edge discrepancy.
  assert np.array_equal(expected,a['x'][row]),row
  idx=np.flatnonzero((t>=edges[-2])&(t<edges[-1]));raw=v[int(idx[0]):int(idx[-1])+1]
  target=(raw.mean(0)*float(v.attrs.get('conversion',1))+float(v.attrs.get('offset',0)))*1000
  err=float(np.max(abs(target-a['y'][row])));max_yerr=max(max_yerr,err);assert err<1e-10,(row,err)
  assert abs(float(a['bin_end'][row])-edges[-1])<1e-9;checked+=1
result={'pass':True,'fitting_rows_independently_rebuilt':checked,'max_target_abs_error':max_yerr,'validation_or_evaluation_arrays_opened':False}
(root/'audit/TRAINING_ROW_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
