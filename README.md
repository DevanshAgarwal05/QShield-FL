# QSHIELD-FL

Hybrid classical + quantum federated learning for network intrusion detection on NSL-KDD,
with a poisoning attack (sign-flip), a robust-aggregation defense (trimmed mean), differential-privacy
noise, and an optional SVD projection in front of the variational quantum circuit (VQC).
A FastAPI dashboard browses saved runs and runs inference on the saved checkpoints.

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt          # add requirements-dev.txt to run the tests
```

Put the NSL-KDD files `KDDTrain+.txt` and `KDDTest+.txt` in `data/nsl_kdd/`
(they are not tracked; processed arrays are cached next to them on first use).

## Run

All commands are run from the repository root.

```
python scripts/main_svd.py --use_svd --ranks 16 --use_20_percent   # SVD hybrid QNN, four FL scenarios
python scripts/main.py                                             # baseline hybrid QNN, four FL scenarios
python scripts/main_classical.py --use_20_percent                  # logreg / MLP / tree baselines
python scripts/draw_qshield_vqc.py                                 # circuit diagram -> docs/qshield_vqc.png
uvicorn backend.app:app                                            # dashboard at http://127.0.0.1:8000
python -m pytest                                                   # unit tests
```

Each FL run writes a `results_*` folder in the repository root (curves, metadata, checkpoints).
Only the reference run `results_svd_hybrid_r16_20260402_030346/` is tracked; other `results_*` folders are ignored.

## Label convention

`data/loader.py` maps **normal -> 1** and **attack -> 0**. Models are trained on that target, so the
model output `sigmoid(logit)` is P(Normal). The API reports `prob_attack = 1 - P(Normal)`, and the batch
confusion matrix treats Attack as the positive class. The constants live in `config.py`.

## Layout

```
backend/    FastAPI app (dashboard API)
models/     HybridQNN, the shared VQC circuit, classical baselines
federated/  client training, aggregation (FedAvg, trimmed mean), attacks, DP noise
data/       NSL-KDD / breast-cancer loaders (raw data is not tracked)
utils/      training loop, SVD projection, checkpoint save/load
web/        dashboard frontend (static)
scripts/    command-line entry points
docs/       reports and the circuit diagram
tests/      unit tests
config.py   paths and shared constants
```
