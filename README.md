# Neural maze decoding

Offline hand-velocity decoding from macaque motor-cortex activity, using the public Jenkins **MC_Maze** recording. A compact temporal convolutional network (TCN) is compared with ridge regression and gradient-boosted trees on later trials from the same session.

## Results

| Model | Mean coordinate R² | RMSE (mm/s) |
|---|---:|---:|
| Three-seed TCN ensemble | 0.834 | 141.3 |
| Ridge, same history | 0.743 | 176.7 |
| Gradient-boosted trees | 0.717 | 185.4 |

The TCN ensemble reduces evaluation MSE by **36.1%** relative to history ridge. Evaluation uses 344 trials / 2,752 samples. It loses to ridge on 42 individual trials; an average advantage is not a guarantee for each movement. R² is not classification accuracy. Full metrics: [`scores/RESULTS.json`](scores/RESULTS.json).

## Method

- **Data:** DANDI 000128, version 0.220113.0400; Jenkins, September 25, 2009. Sorted spike timestamps from 90 M1 units and recorded hand velocity. See [data attribution](DATA_ATTRIBUTION.md).
- **Examples:** 50-ms spike-count bins; each input is **7 × 90**, covering 350 ms. The target is horizontal and vertical hand velocity averaged within the final bin, not a future-motion forecast.
- **Sampling:** eight overlapping windows per trial, drawn from a 700-ms segment aligned to annotated movement onset. Windows never cross trials.
- **Split:** chronological whole-trial 60/20/20 subdivision of the official training split, with a two-second boundary purge: 1,032 fit, 343 validation, and 344 evaluation trials. This is not the official benchmark test.
- **Network:** width-3 convolution with dilation 1 → ReLU → width-3 convolution with dilation 2 → ReLU → linear two-coordinate readout. Selected models have 32 hidden channels. Three seeds are averaged.
- **Training:** square-root count transform and fitting-only scaling; Adam, minibatches of 128, up to 40 epochs. Architecture, weight penalty, and checkpoint are selected on validation MSE; final evaluation does not select models.

## Run

Python 3.12; the requirements pin the original numerical environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python check_project.py --output ../maze-check
python -m unittest test_inference_release.py
```

The check verifies package integrity, model artifact hashes, all nine saved prediction streams, and published metrics. It does **not** retrain or test new data. Saved inference was also reported to reproduce on macOS; full retraining parity was established in the recorded Linux environment, not every platform.

For one label-free prediction:

```bash
python infer.py --input example/neural_input.npz --output ../prediction.npz
```

For complete retraining from prepared arrays:

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reproduce.py ../maze-reproduction
```

Outputs must not already exist. See [reproduction instructions](QUICKSTART.md) for raw-data download and independent reconstruction. Only load trusted pickle models: Python pickle can execute code.

## Scope and limitations

One animal, one session, selected successful trials, and familiar maze conditions. Movement-onset alignment is supplied offline, source velocity processing may be noncausal, and same-bin counts become available only at bin end. This is **not** continuous online decoding, cross-day/cross-animal transfer, closed-loop control, or clinical validation. Repeating this experiment reproduces exposed evidence; it is not a new held-out test.

## Repository map

- `model.py`, `engine.py`, `study.py`: network, fitting, selection, evaluation.
- `prepare.py`, `download_raw.py`: preparation and pinned source acquisition.
- `data/`, `models/`, `scores/`: prepared arrays, fitted models, predictions and metrics.
- `audit/`, `source/`, `SOURCE.json`: verification and dataset provenance.
- `PUBLIC_PACKAGE.json`: packaging changes and preservation checks.

This public distribution omits the original historical protocol and pre-registration seal chain. Numerical model code, prepared data, fitted parameters, predictions and metrics are unchanged; acquisition and verification wrappers use the public manifests. Current checks verify file integrity and numerical reproducibility, **not the historical timing of pre-registration or model selection**. Historical audit reports describe the original experiment, not new runs of this distribution.

## Licensing

Source recordings and distributed derived data: **CC BY 4.0**, with attribution in [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). No separate software license grant is supplied; do not infer that the data license covers the analysis code. This repository is not affiliated with Neuralink.
