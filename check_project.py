"""Verify this offline project and replay saved inference. No training or network."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,platform,sys
from datetime import datetime,timezone
import numpy as np
import infer

ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def verify_distribution():
    manifest=read(ROOT/'PACKAGE_MANIFEST.json')
    for name,record in manifest['files'].items():
        p=ROOT/name
        if not p.is_file() or p.stat().st_size!=record['bytes'] or digest(p)!=record['sha256']:
            raise ValueError('Distributed file changed or missing: '+name)
    return len(manifest['files'])

def check(output):
    output=Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Choose a new report directory; checks never overwrite earlier runs')
    count=verify_distribution()
    import study
    study.verify_models(ROOT/'models')
    # The data were already exposed in the original experiment. Truth is used
    # only to verify published arrays/metrics, never passed to predict_file.
    with np.load(ROOT/'data/evaluation.npz',allow_pickle=False) as d:
        x=d['x'].copy();units=d['unit_ids'].copy();y=d['y'].copy()
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'neural_counts_only.npz',x=x,unit_ids=units)
    inference=infer.predict_file(output/'neural_counts_only.npz',output/'replayed_predictions.npz')
    published=read(ROOT/'scores/RESULTS.json')
    errors={};metric_errors={}
    with np.load(output/'replayed_predictions.npz',allow_pickle=False) as actual,np.load(ROOT/'scores/predictions.npz',allow_pickle=False) as expected:
        if set(actual.files)!=set(published['metrics']):raise AssertionError('Missing or extra model streams')
        np.testing.assert_array_equal(y,expected['y'])
        for name in actual.files:
            np.testing.assert_allclose(actual[name],expected[name],rtol=1e-10,atol=1e-8)
            errors[name]=float(np.max(np.abs(actual[name]-expected[name])))
            measured=study.e.metric(y,actual[name]);ref=published['metrics'][name]
            for key in ['mse','rmse','mean_r2','r2_per_axis','rows']:
                np.testing.assert_allclose(measured[key],ref[key],rtol=1e-10,atol=1e-8)
            metric_errors[name]=abs(measured['mse']-ref['mse'])
    # Recheck all distributed hashes after execution as well.
    if verify_distribution()!=count:raise AssertionError('Distribution changed')
    result=dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),
        verification_scope='Packaged file integrity, model artifact hashes, full saved-inference replay and published metrics; not a new held-out experiment or full retraining.',
        distributed_files_checked=count,models_replayed=len(errors),evaluation_rows=len(x),
        max_prediction_difference=max(errors.values()),model_prediction_differences=errors,
        max_mse_difference=max(metric_errors.values()),truth_supplied_to_model=False,
        models_retrained=False,network_access=False,
        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn']},
        python=platform.python_version(),platform=platform.system(),
        remaining_limits=['One subject/session, source-selected successful trials, offline movement-onset alignment.',
            'New-source training, not zero-shot cross-animal transfer or online/clinical control.',
            'This run verifies saved inference on the reported platform, not full retraining or portability to other platforms.'])
    (output/'RUN_REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New directory for predictions and RUN_REPORT.json')
    args=parser.parse_args(argv)
    try:result=check(args.output)
    except (ValueError,OSError,AssertionError) as error:parser.error(str(error))
    print(json.dumps(result,indent=2));print('Report:',args.output/'RUN_REPORT.json')

if __name__=='__main__':main()
