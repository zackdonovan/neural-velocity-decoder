"""Trial-aware preparation fixed before fitting, with no outcome-based choices."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parent
DT=.05

def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def ragged(group,key,index):
    end=int(group[key+'_index'][index]);start=int(group[key+'_index'][index-1]) if index else 0
    return group[key][start:end]
def merge_intervals(a):
    a=np.asarray(a,float)
    if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all() or (a[:,1]<a[:,0]).any():raise ValueError('Invalid observation intervals')
    a=a[np.argsort(a[:,0],kind='stable')];out=[]
    for lo,hi in a:
        if out and lo<=out[-1][1]+1e-9:out[-1][1]=max(out[-1][1],hi)
        else:out.append([lo,hi])
    return np.asarray(out)
def covered(intervals,lo,hi):
    i=np.searchsorted(intervals[:,0],lo+1e-9,side='right')-1
    return (i>=0)&(intervals[np.maximum(i,0),1]>=hi-1e-9)
def spike_counts(times,edges):return np.diff(np.searchsorted(times,edges,side='left'))
def behavior_bin(times,values,lo,hi,nominal):
    a,b=np.searchsorted(times,[lo,hi],side='left')
    if a==b:return None
    t=times[a:b];v=np.asarray(values[a:b])
    if t[0]>lo+1.01*nominal or t[-1]<hi-1.01*nominal:return None
    if len(t)>1 and np.max(np.diff(t))>2*nominal+1e-12:return None
    if v.shape!=(len(t),2) or not np.isfinite(v).all():return None
    return v.mean(0)

def prepare():
    p=json.loads((ROOT/'SOURCE.json').read_text())
    raw=ROOT/'raw'/Path(p['source_path']).name
    if raw.stat().st_size!=p['source_bytes'] or digest(raw)!=p['source_sha256']:raise ValueError('Wrong source bytes')
    output=ROOT/'data';output.mkdir(exist_ok=False)
    reports=[];splits={};unit_ids=None
    with h5py.File(raw,'r') as f:
        tr=f['intervals/trials'];units=f['units']
        starts,stops,ids,moves=tr['start_time'][:],tr['stop_time'][:],tr['id'][:],tr['move_onset_time'][:]
        labels=np.array([x.decode() if isinstance(x,bytes) else str(x) for x in tr['split'][:]])
        eligible=np.flatnonzero(labels=='train');eligible=eligible[np.lexsort((ids[eligible],starts[eligible]))]
        if not len(eligible) or not np.isfinite(starts[eligible]).all() or not np.isfinite(stops[eligible]).all():raise ValueError('Invalid trial metadata')
        if (stops[eligible]<=starts[eligible]).any() or (starts[eligible][1:]<stops[eligible][:-1]-1e-9).any():raise ValueError('Overlapping or malformed trials; stop rather than revise split')
        n=len(eligible);cut1=int(np.floor(.6*n));cut2=cut1+int(np.floor(.2*n))
        nominal={'fit':eligible[:cut1],'validation':eligible[cut1:cut2],'evaluation':eligible[cut2:]}
        before={'fit':None,'validation':float(stops[eligible[cut1-1]]),'evaluation':float(stops[eligible[cut2-1]])}
        allids=units['id'][:];cols=np.array([j for j,x in enumerate(allids) if str(int(x)).startswith('2')])
        unit_ids=allids[cols]
        if len(cols)<2 or len(np.unique(unit_ids))!=len(unit_ids):raise ValueError('Invalid M1 unit identity')
        spikes=[ragged(units,'spike_times',int(j)) for j in cols]
        observations=[merge_intervals(ragged(units,'obs_intervals',int(j))) for j in cols]
        for a in spikes:
            if not np.isfinite(a).all() or (np.diff(a)<0).any():raise ValueError('Invalid spike ordering')
        behavior=f['processing/behavior/hand_vel'];t=behavior['timestamps'][:];values=behavior['data']
        delta=np.diff(t)
        if not np.isfinite(t).all() or not (delta>0).all():raise ValueError('Invalid behavior timestamps')
        nominal_dt=float(np.median(delta))
        gaps=np.flatnonzero(delta>2*nominal_dt)
        if len(gaps):
            if len(gaps)!=len(starts)-1 or not np.allclose(t[gaps]+nominal_dt,stops[:-1],rtol=0,atol=1e-8) or not np.allclose(t[gaps+1],starts[1:],rtol=0,atol=1e-8):
                raise ValueError('Unexpected gaps within trials; stop under fixed coverage rule')
        within=delta<=2*nominal_dt
        if np.max(np.abs(delta[within]-nominal_dt))>nominal_dt*.01:raise ValueError('Irregular within-trial sampling')
        unit=values.attrs['unit'];unit=unit.decode() if isinstance(unit,bytes) else str(unit)
        if unit!='m/s' or values.shape!=(len(t),2):raise ValueError('Unexpected velocity units/schema')
        conversion=float(values.attrs.get('conversion',1));offset=float(values.attrs.get('offset',0))
        if not np.isfinite(conversion) or not np.isfinite(offset):raise ValueError('Invalid physical-unit conversion')
        conditions={key:tr[key][:] for key in ['maze_id','trial_version']}
        for role,trial_rows in nominal.items():
            xs=[];ys=[];source_rows=[];trialids=[];bins=[];ends=[];sc=[];used_trials=[]
            for row in trial_rows:
                row=int(row);record=dict(split=role,source_trial_row=row,trial_id=int(ids[row]),start=float(starts[row]),stop=float(stops[row]),maze_id=int(conditions['maze_id'][row]),trial_version=int(conditions['trial_version'][row]),valid_target_bins=[],reason=None)
                if before[role] is not None and starts[row]<before[role]+2:
                    record['reason']='Two-second split-boundary purge';reports.append(record);continue
                if not np.isfinite(moves[row]):
                    record['reason']='No finite move onset';reports.append(record);continue
                edges=moves[row]+(-.25+np.arange(15)*DT)
                good=(edges[:-1]>=starts[row]-1e-9)&(edges[1:]<=stops[row]+1e-9)
                counts=np.empty((14,len(cols)),np.int32)
                for j,(spike,obs) in enumerate(zip(spikes,observations)):
                    counts[:,j]=spike_counts(spike,edges);good&=covered(obs,edges[:-1],edges[1:])
                sc.extend(counts[good])
                for k in range(6,14):
                    if not good[k-6:k+1].all():continue
                    y=behavior_bin(t,values,edges[k],edges[k+1],nominal_dt)
                    if y is None:continue
                    y=(y*conversion+offset)*1000 # Report mm/s; honors NWB SI conversion.
                    xs.append(counts[k-6:k+1]);ys.append(y);source_rows.append(row);trialids.append(ids[row]);bins.append(k);ends.append(edges[k+1]);record['valid_target_bins'].append(k)
                if record['valid_target_bins']:used_trials.append(row)
                else:record['reason']='No valid history/target rows under fixed coverage rule'
                reports.append(record)
            if not xs:raise ValueError('No valid data for '+role)
            arrays=dict(x=np.asarray(xs,dtype=np.int32),y=np.asarray(ys),counts=np.asarray(sc,dtype=np.int32),
                        source_trial_row=np.asarray(source_rows),trial_id=np.asarray(trialids),bin_index=np.asarray(bins),bin_end=np.asarray(ends),unit_ids=unit_ids)
            np.savez_compressed(output/(role+'.npz'),**arrays)
            splitinfo=dict(nominal_trial_count=len(trial_rows),valid_trial_count=len(used_trials),rows=len(xs),source_trial_rows=trial_rows.tolist(),used_trial_rows=used_trials,
                           neural_window_seconds=len(used_trials)*.7,target_seconds=len(xs)*DT,nominal_start=float(starts[trial_rows[0]]),nominal_stop=float(stops[trial_rows[-1]]),
                           source_conditions=sorted({(int(conditions['maze_id'][i]),int(conditions['trial_version'][i])) for i in used_trials}),sha256=digest(output/(role+'.npz')))
            splits[role]=splitinfo
        thresholds={'fit':(100,1000),'validation':(30,200),'evaluation':(30,200)}
        for role,(nt,nr) in thresholds.items():
            if splits[role]['valid_trial_count']<nt or splits[role]['rows']<nr:raise ValueError('Fixed eligibility threshold failed: '+role)
        overlap={a+'__'+b:len({tuple(x) for x in splits[a]['source_conditions']} & {tuple(x) for x in splits[b]['source_conditions']}) for a,b in [('fit','validation'),('fit','evaluation'),('validation','evaluation')]}
        meta=dict(source_sha256=p['source_sha256'],official_train_trials=n,excluded_source_val_trials=int((labels=='val').sum()),unit_ids=unit_ids.tolist(),input_units=len(unit_ids),velocity_native_output_units='mm/s',nwb_unit=unit,nwb_conversion=conversion,nwb_offset=offset,nominal_behavior_dt=nominal_dt,splits=splits,condition_overlap=overlap,prepared_utc=datetime.now(timezone.utc).isoformat(),scope='Mechanical preparation includes final target values, but no final outcome statistics, model selection or scoring is performed.')
        dump(output/'metadata.json',meta);dump(output/'TRIAL_LEDGER.json',reports)
        print(json.dumps({k:v for k,v in meta.items() if k in ['official_train_trials','excluded_source_val_trials','input_units','condition_overlap']},indent=2))
        print(json.dumps({k:{q:v[q] for q in ['nominal_trial_count','valid_trial_count','rows']} for k,v in splits.items()},indent=2))
if __name__=='__main__':prepare()
