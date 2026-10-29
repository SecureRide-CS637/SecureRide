# SecureRide model + dual-mode detection (add-on to the dataset)

Drop these files into the dataset folder (next to `secureride_data.py` and `data/`).

| file | what it does |
|---|---|
| `secureride_model.py` | ResNet-based 1D CNN (attention block + HL residual blocks), GRU baseline, and a scikit-learn fallback, all behind one `Predictor` API |
| `train.py` | trains the HL (1-3) x SL (5-30) grid for both tasks, writes `results/accuracy_<backend>.csv` and the Fig. 5-style plot |
| `dualmode.py` | Algorithm 1 (streaming state machine), model selection from tau / alpha (Eq. 5, 6), 9-case evaluation on the scenario trips, Fig. 9-style plots |
| `calibration_plots.py` | Fig. 2a and 2b equivalents plus a printed summary of the numbers |

## Setup
    pip install numpy pandas matplotlib scikit-learn joblib
    pip install tensorflow==2.13.*          # only for the real CNN (paper used TF 2.13)
    python3 secureride_data.py build --root data     # only if data/secureride_windows.npz is missing

## Run order
    python3 calibration_plots.py --root data --out results
    python3 train.py --backend keras                  # real CNN, full grid (use a GPU/Colab if you can)
    python3 train.py --backend keras --arch gru       # optional GRU baseline
    python3 dualmode.py --task sidewalk --backend keras
    python3 dualmode.py --task speeding --backend keras

Quick no-TensorFlow run (what produced the files in `results/`):
    python3 train.py --backend sklearn --max-train 40000 --max-eval 40000

Useful flags: `--lengths 5 10 20`, `--hls 1`, `--epochs 30`, `--no-augment`, `--l2 0.01`,
`dualmode.py --taus 1 1.5 2 --alphas 0.9 0.95 0.99 --use-history`.

## What follows the paper
- Input is only voltage, current, capacity. Speed, surface and labels are never model inputs.
- Training data = Gaussian U spike U step/slope augmentation (+ originals), train split only.
- CNN: attention block, up to 3 residual blocks of 3 Conv1D (BN, ReLU, dropout 0.3), global average pooling, softmax. Adam, categorical cross-entropy, ReduceLROnPlateau, early stopping patience 10.
- Dual mode = Algorithm 1: Quick by default; on a detection switch to Accurate; if Accurate says legal, go back to Quick. The first Accurate window needs L_long - L_short new samples, so the Quick verdict is held meanwhile.
- Quick model: HL 1, longest L with `L*0.2 s + exec time <= tau`. Accurate model: fewest-slot model with HL <= 2 reaching `alpha`. Chosen on validation accuracy, reported on the scenario trips.

## Deviations to state in your report
- L2 default is 1e-3 (paper: 0.01, which tends to underfit a model this small). `--l2 0.01` is literal.
- Alpha defaults are spread over this dataset's achievable accuracy; the paper's 93-99 % assume its real data. Override with `--alphas`.
- Detection time = (slots from illegal onset to the first run of 3 consecutive detections) x 0.2 s + execution time measured on YOUR machine, not on an MCU. It can be below L x 0.2 s because a window that is only partly illegal can already trigger a (noisy) Quick detection.
- `exec_ms` in the csv is a laptop number; the paper's 9.3 ms-605 ms are Arduino numbers.

## Results already in `results/` (sklearn fallback, NOT the CNN)
Test accuracy rises with sequence length (sidewalk 77 % at SL5 to 88 % at SL30; speeding 79 % to 87 %). More hidden layers do not help here because HL only changes tree count/depth for this backend; that is expected and NOT a CNN result. Re-run with `--backend keras` for the real HL trend.
Dual mode, sidewalk: ~454-472 correct slots out of 500 vs 407-422 (Quick) and 437-454 (Accurate), with detection time 0.3-0.6 s vs 1.3-3.3 s for Accurate. Speeding: Dual is roughly equal to Quick (the speeding labels flicker near 20 km/h, and Accurate is only slightly better than Quick), so do not claim a dual-mode gain there unless the CNN shows it.

---
# Contributions added on top of the paper (limitations the paper names)

## 1. Class imbalance  -> `imbalance.py`  (+ `class_weight` option in every backend of `secureride_model.py`)
Rebuilds windows from the raw trips at 50 % / 5 % / 1 % illegal share (train, val and test all use that share),
trains with and without class weights, and reports precision, recall, F1, false-positive rate, average precision and
the accuracy of an "always legal" classifier. Extra rows `+thr` tune the decision threshold on the validation set.
    python3 imbalance.py --backend torch --task sidewalk --L 10 --hl 2        # your CNN (use the HL/SL you picked)
    python3 imbalance.py --backend torch --task speeding --L 20 --hl 2
    python3 imbalance.py --backend sklearn --task sidewalk --L 10 --hl 1      # quick test, no PyTorch needed
Outputs: `results/imbalance_<tag>.csv`, `..._summary.csv`, `results/fig_imbalance_<tag>.png`.

## 2. Temporal smoothing  -> `smoothing.py`  (+ `majority_vote`, `MajorityVote` in `dualmode.py`)
Causal majority vote over the last N verdicts (N = 1 is the paper). `smoothing.py` sweeps N on all held-out test trips
(false-alarm events per hour, recall, precision, F1) and on the 100 s scenarios (detection time, correct slots).
    python3 smoothing.py --task sidewalk --backend torch --quick-L 5 --quick-hl 1 --acc-L 20 --acc-hl 2
    python3 smoothing.py --task speeding --backend torch
    python3 dualmode.py  --task sidewalk --backend torch --smooth-n 5         # smoothing inside the 9-case dual-mode study
Outputs: `results/smoothing_<backend>_<task>.csv`, `results/fig_smoothing_<backend>_<task>.png`.

## Tests
    python3 test_contributions.py        # 10 tests, no PyTorch needed

## Fix in `dualmode.py`
`evaluate_case` now loads `<stem>.<ext of the chosen backend>`. Before, `load_predictor(stem)` returned the `.pt` file
first even with `--backend sklearn` whenever both `.pt` and `.joblib` files were present.

## Results in `results/` for these two contributions are from the sklearn fallback (PyTorch was not available when they were produced).
Re-run the commands above with `--backend torch` and use THOSE numbers for the CNN in your slides.
