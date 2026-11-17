"""Run saved models on counts and unit IDs only; never overwrite predictions."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import io
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def validate_output_path(output):
    output=Path(output)
    if output.suffix.lower()!='.npz':
        raise ValueError('Output must end with .npz, for example my_prediction.npz')
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output already exists; choose a new file: {output}')
    if not output.parent.is_dir():
        raise FileNotFoundError(f'Output directory does not exist: {output.parent}')
    return output


def validate_input_arrays(x,units,expected):
    x=np.asarray(x);units=np.asarray(units);expected=np.asarray(expected)
    if x.ndim!=3 or x.shape[1:]!=(7,len(expected)) or not len(x):
        raise ValueError(f'Expected a nonempty N by 7 by {len(expected)} count array')
    if not np.array_equal(units,expected):
        raise ValueError('Wrong unit identity or ordering')
    if x.dtype.kind not in 'biuf' or not np.isfinite(x).all() or (x<0).any() or not np.equal(x,np.floor(x)).all():
        raise ValueError('Inputs must be finite nonnegative integer spike counts')
    return x


def write_exclusive_npz(output,arrays):
    """Serialize before creating the exclusive file; never append a suffix.

    Exclusive creation protects an existing file, including a destination
    created after the initial check. A caught write error removes our partial
    output. A forcibly killed process can still leave a partial file; do not
    treat the absence of a success message as a completed prediction.
    """
    output=validate_output_path(output)
    payload=io.BytesIO()
    np.savez_compressed(payload,**arrays)
    # Passing an opened file, rather than its name, disables NumPy's implicit
    # '.npz' suffix addition. Mode xb is a second, race-safe no-overwrite check.
    with output.open('xb') as stream:
        created_stat=os.fstat(stream.fileno())
        try:
            stream.write(payload.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            # Do not remove a different file substituted after we created ours.
            stream.close()
            try:
                now=output.lstat()
                if (now.st_dev,now.st_ino)==(created_stat.st_dev,created_stat.st_ino):
                    output.unlink()
            except FileNotFoundError:
                pass
            raise
    return output


def predict_file(input_path,output,models=None):
    output=validate_output_path(output)
    import study
    models=ROOT/'models' if models is None else Path(models)
    study.verify_models(models)
    expected=np.array(json.loads((ROOT/'data/metadata.json').read_text())['unit_ids'])
    with np.load(input_path,allow_pickle=False) as source:
        if not {'x','unit_ids'}.issubset(source.files):
            raise ValueError('Input must contain x and unit_ids arrays')
        x=validate_input_arrays(source['x'],source['unit_ids'],expected)
    answer=study.predict({'x':x},models)
    if not answer or any(a.shape!=(len(x),2) or not np.isfinite(a).all() for a in answer.values()):
        raise ValueError('Model returned invalid predictions; nothing was written')
    write_exclusive_npz(output,answer)
    return dict(rows=len(x),units='mm/s',model_outputs=list(answer),truth_used=False,output=str(output))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New .npz file; existing files are never replaced')
    parser.add_argument('--models',type=Path,default=ROOT/'models',help='Trusted model folder only')
    args=parser.parse_args(argv)
    try:
        result=predict_file(args.input,args.output,args.models)
    except (ValueError,OSError) as error:
        parser.error(str(error))
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    main()
