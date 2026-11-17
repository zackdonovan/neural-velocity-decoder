"""Read structural timestamps/trial boundaries only; never access behavior values."""
from pathlib import Path
import h5py,numpy as np,json
root=Path(__file__).resolve().parents[1]
with h5py.File(root/'raw/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb','r') as f:
    t=f['processing/behavior/hand_vel/timestamps'][:]
    tr=f['intervals/trials']; starts=tr['start_time'][:];stops=tr['stop_time'][:]
    order=np.argsort(starts,kind='stable');starts=starts[order];stops=stops[order]
    delta=np.diff(t);dt=np.median(delta);g=np.flatnonzero(delta>2*dt)
    assert np.all(np.isfinite(t)) and np.all(delta>0)
    assert len(g)==len(starts)-1
    assert np.all(starts[1:]>=stops[:-1]-1e-9)
    assert np.allclose(t[g],stops[:-1]-dt,atol=1e-9,rtol=0)
    assert np.allclose(t[g+1],starts[1:],atol=1e-9,rtol=0)
    assert np.max(np.abs(delta[delta<=2*dt]-dt))<1e-9
    assert abs(t[0]-starts[0])<1e-9 and abs(t[-1]-(stops[-1]-dt))<1e-9
    result={'pass':True,'source_trials':len(starts),'intertrial_gaps':len(g),'nominal_dt':float(dt),'max_within_trial_step_error':float(np.max(np.abs(delta[delta<=2*dt]-dt))),'behavior_values_accessed':False}
(root/'audit/TIMESTAMP_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
