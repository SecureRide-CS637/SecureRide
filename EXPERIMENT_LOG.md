# SecureRide: Experiment and Repository Snapshot

**Snapshot reviewed:** November 1, 2026
**Scope:** Current top-level simulator, training, inference, and evaluation code. Metrics below are from local generated CSVs; they are not a fresh rerun of every experiment.

## 1. Current implementation

The active workflow in this repository is:

```text
gen_dataset.py
    -> data/trips.csv, data/trips_meta.csv
secureride_data.py build
    -> data/secureride_windows.npz, data/norm_stats.json, data/window_counts.csv
train.py
    -> models/, results/metrics_*.csv, confusion matrices and plots
dualmode.py / predict_sample.py
    -> scenario evaluation / sample-file inference
```

`train.py` and `dualmode.py` support PyTorch, Keras, and scikit-learn backends; CNN and GRU architectures are available for the neural backends. `imbalance.py` evaluates alternate class prevalences, and `smoothing.py` evaluates causal majority-vote smoothing.

The current training flow reads `data/trips.csv` through `secureride_data.py`. The checked-in `data/raw/`, `data/processed/`, and `data/final/` directories are not inputs to these active scripts. Likewise, `models/theta_*.h5`, `models/theta_*.tflite`, and the manifest describe a separate 1 Hz / voltage-current-SoC model artifact; the top-level training flow uses 5 Hz windows with voltage-current-capacity channels. Do not mix their metrics or preprocessing.

Generated files under `results/` are ignored by Git, so the CSVs cited below may not be present in a fresh clone. Regenerate them with the commands in Section 6 before relying on them.

## 2. Dataset and simulator checks

- The current `data/trips.csv` contains **668,003 rows** across **760 trip IDs**.
- The non-scenario split contains **460 train**, **116 validation**, and **144 test** trips. There are also 20 scenario trips per task.
- Telemetry is simulated at **5 Hz**. Inputs are voltage, current, and capacity; speed and surface are used for targets/analysis, not as model input channels.
- The prepared window labels use the all-points rule: a window is positive only if all of its timesteps are labeled illegal. Windows are balanced by task, length, and split; the imbalance experiment deliberately constructs other prevalences.
- `gen_dataset.py` samples steady-state speed targets from `(1, 5)`, `(5, 10)`, `(10, 15)`, `(15, 19)`, and `(21.5, 24.5)` km/h. The resulting 19–21.5 km/h target gap is a simulator design choice. In the current trips file, **38,640 rows (5.78%)** have measured speeds strictly inside that interval.

### Calibration results

The local `results/calibration_summary.txt` records:

- Voltage drop at speeds $\geq 20$ km/h versus below 20 km/h: **2.72%**; capacity-quintile-controlled drops: **2.50–3.39%** (mean **2.82%**).
- Standard deviation of first-difference voltage: **0.391 V** on the simulated road and **0.910 V** across simulated sidewalk surfaces, a **2.33×** ratio.

These are simulator outputs, not field measurements. The difference from paper-reported values has not been resolved into a single causal explanation.

## 3. Recorded test-set model results

Selected rows from the local `results/metrics_torch.csv`, `results/metrics_torch_gru.csv`, and `results/metrics_sklearn.csv` files:

| Task | Model | Backend | L | HL | Test accuracy | Test F1 | Balanced accuracy | Inference ms | Size KB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Speeding | CNN | PyTorch | 20 | 2 | 83.53% | 0.8357 | 0.8355 | 0.249 | 63.3 |
| Sidewalk | CNN | PyTorch | 20 | 2 | 88.00% | 0.8764 | 0.8799 | 0.244 | 63.3 |
| Speeding | GRU | PyTorch | 20 | 2 | 82.67% | 0.8263 | 0.8269 | 0.399 | 43.1 |
| Sidewalk | GRU | PyTorch | 20 | 2 | 84.33% | 0.8273 | 0.8432 | 0.403 | 43.1 |
| Speeding | HistGradientBoosting | sklearn | 20 | 1 | 86.51% | 0.8672 | 0.8650 | 0.929 | 98.7 |
| Sidewalk | HistGradientBoosting | sklearn | 30 | 1 | 87.77% | 0.8715 | 0.8776 | 1.046 | 98.9 |

These are selected configurations from one local artifact set, not universal model rankings. Inference times depend on the machine and measurement conditions; the CSVs do not document enough hardware detail for cross-platform timing comparisons.

## 4. Recorded dual-mode results

The local `results/dualmode_sidewalk_torch.csv` and `results/dualmode_speeding_torch.csv` each contain nine cases across $\tau \in \{1.0, 1.5, 2.0\}$ seconds and three validation-accuracy targets.

| Task | Case | Alpha | Quick model | Accurate model | Quick correct / 500 | Accurate correct / 500 | Dual correct / 500 | Quick detect (s) | Accurate detect (s) | Dual detect (s) |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| Sidewalk | 1 | 0.835 | SL5/HL1 | SL10/HL1 | 380.9 | 426.2 | 441.8 | 1.120 | 1.540 | 0.570 |
| Sidewalk | 2 | 0.875 | SL5/HL1 | SL20/HL1 | 380.9 | 440.2 | 465.3 | 1.120 | 3.090 | 0.570 |
| Speeding | 1 | 0.823 | SL5/HL1 | SL10/HL2 | 454.4 | 460.7 | 455.8 | 0.495 | 2.221 | 0.495 |
| Speeding | 2 | 0.846 | SL5/HL1 | SL20/HL2 | 454.4 | 455.6 | 443.1 | 0.495 | 4.090 | 0.516 |

These configurations repeat across $\tau \in \{1.0,1.5,2.0\}$ seconds. The recorded sidewalk cases improve slot correctness over the listed fixed models, while the speeding dual-mode cases do not. This supports describing dual mode as an accuracy/latency tradeoff, not as a guaranteed accuracy improvement.

## 5. Tests and sample inference

The repository includes:

- `test_secureride.py`: dataset, labeling, split isolation, augmentation, model serialization, sample-input validation, dual-mode transitions, and metric tests.
- `test_contributions.py`: imbalance-weighting and temporal-smoothing tests.

Run them with:

```bash
python -m unittest test_secureride.py
python test_contributions.py
```

The earlier experiment notes recorded 12 tests passing on a Windows/Python 3.11 environment. That historical result is not a guarantee for other dependency versions or hosts. The current dependency file gives minimum versions rather than a fully locked environment.

The sample CSVs in the repository can be passed to `predict_sample.py`. The test suite checks their required columns and minimum length; prediction results should be recorded against the exact model/backend used.

## 6. Reproduction commands

From the repository root:

```bash
python gen_dataset.py
python secureride_data.py build --root data
python train.py --backend sklearn --tasks speeding sidewalk \
  --lengths 5 10 20 30 --hls 1 2 3 \
  --max-train 15000 --max-eval 10000
python train.py --backend torch --arch cnn --tasks speeding sidewalk \
  --lengths 5 10 20 --hls 1 2 \
  --max-train 4000 --max-eval 3000 --epochs 15 --batch 256
python train.py --backend torch --arch gru --tasks speeding sidewalk \
  --lengths 5 10 20 --hls 1 2 \
  --max-train 4000 --max-eval 3000 --epochs 15 --batch 256
python dualmode.py --task sidewalk --backend torch
python -m unittest test_secureride.py
python test_contributions.py
```

To run the scikit-learn imbalance experiment or smoothing sweep, see `imbalance.py --help` and `smoothing.py --help`. Record backend, seed, model configuration, and output filenames with any new result because training and timing outputs can vary by run and host.

## 7. Limitations

1. The generated telemetry and scenario labels are synthetic. The repository does not validate detection on independent real scooter rides.
2. The simulator's speed-target gap differs from the paper's continuous speed bins.
3. Accuracy, calibration, and dual-mode scores depend on simulator parameters, trip splits, and model configuration.
4. Local inference timing is not an embedded-device benchmark.
5. Battery measurements are not demonstrated to be tamper-proof; resistance to sensor spoofing or manipulation is outside the evaluated scope.
