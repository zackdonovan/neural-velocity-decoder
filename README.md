# neural velocity decoder

hand-velocity decoding from macaque motor-cortex activity using the public jenkins MC_Maze recording. compares a small temporal convolutional network (TCN), ridge regression and gradient-boosted trees on later trials from the same session. this is an offline experiment.

## results

| model | mean coordinate R² | RMSE (mm/s) |
|---|---:|---:|
| three-seed TCN ensemble | 0.834 | 141.3 |
| ridge, same history | 0.743 | 176.7 |
| gradient-boosted trees | 0.717 | 185.4 |

the TCN ensemble reduces evaluation MSE by 36.1% relative to history ridge. evaluation uses 344 trials / 2,752 samples. it loses to ridge on 42 individual trials. R² measures regression fit, not classification accuracy. full metrics: [`scores/RESULTS.json`](scores/RESULTS.json).

## method

- data: DANDI 000128, version 0.220113.0400; jenkins, september 25, 2009. sorted spike timestamps from 90 M1 units and recorded hand velocity. see [data attribution](DATA_ATTRIBUTION.md).
- examples: 50-ms spike-count bins; each input is 7 × 90, covering 350 ms. the target is horizontal and vertical hand velocity averaged within the final bin, not a future-motion forecast.
- sampling: eight overlapping windows per trial, drawn from a 700-ms segment aligned to annotated movement onset. windows never cross trials.
- split: chronological whole-trial 60/20/20 subdivision of the official training split, with a two-second boundary purge: 1,032 fit, 343 validation, and 344 evaluation trials. this is not the official benchmark test.
- network: width-3 convolution with dilation 1 → ReLU → width-3 convolution with dilation 2 → ReLU → linear two-coordinate readout. selected models have 32 hidden channels. three seeds are averaged.
- training: square-root count transform and fitting-only scaling; Adam, minibatches of 128, up to 40 epochs. architecture, weight penalty, and checkpoint are selected on validation MSE; final evaluation does not select models.

## run

Python 3.12; the requirements pin the original numerical environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python check_project.py --output ../maze-check
python -m unittest test_inference_release.py
```

the check verifies package integrity, model artifact hashes, all nine saved prediction streams, and published metrics. it does not retrain or test new data. saved inference was also reported to reproduce on macOS; full retraining parity was established in the recorded Linux environment, not every platform.

for one label-free prediction:

```bash
python infer.py --input example/neural_input.npz --output ../prediction.npz
```

for complete retraining from prepared arrays:

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reproduce.py ../maze-reproduction
```

outputs must not already exist. see [reproduction instructions](QUICKSTART.md) for raw-data download and independent reconstruction. only load trusted pickle models: Python pickle can execute code.

## scope and limitations

one animal, one session, selected successful trials, and familiar maze conditions. movement-onset alignment is supplied offline, source velocity processing may be noncausal, and same-bin counts become available only at bin end. this is not continuous online decoding, cross-day/cross-animal transfer, closed-loop control, or clinical validation. repeating this experiment reproduces exposed evidence; it is not a new held-out test.

## files

- `model.py`, `engine.py`, `study.py`: network, fitting, selection, evaluation.
- `prepare.py`, `download_raw.py`: preparation and pinned source acquisition.
- `data/`, `models/`, `scores/`: prepared arrays, fitted models, predictions and metrics.
- `audit/`, `source/`, `SOURCE.json`: verification and dataset provenance.
- `PUBLIC_PACKAGE.json`: packaging changes and preservation checks.

this public distribution omits the original historical protocol and pre-registration seal chain. numerical model code, prepared data, fitted parameters, predictions and metrics are unchanged; acquisition and verification wrappers use the public manifests. current checks verify file integrity and numerical reproducibility, not the historical timing of pre-registration or model selection. historical audit reports describe the original experiment, not new runs of this distribution.

## licensing

source recordings and distributed derived data: CC BY 4.0, with attribution in [data_attribution.md](DATA_ATTRIBUTION.md). no separate software license grant is supplied; do not infer that the data license covers the analysis code. this repository is not affiliated with Neuralink.
