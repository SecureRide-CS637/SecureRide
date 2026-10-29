# SecureRide: Detecting Safety-Threatening Behavior of E-Scooters Using Battery Information

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%2Bcpu-EE4C2C.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An academic, simulation-based implementation and evaluation of **SecureRide** (ACM TECS / ACM EMSOFT 2025). This project uses internal battery telemetry (voltage, current, remaining capacity) as a tamper-resistant physical side channel to detect **speeding violations** and **illegal sidewalk riding** on electric scooters without relying on vulnerable IMUs or degraded urban GPS signals.

---

## ⚡ Quick Start: 3 Commands to Demonstration

```powershell
# 1. Run Automated Test Suite
python -m unittest test_secureride.py

# 2. Run Sample Prediction on Synthetic Telemetry Traces
python predict_sample.py --input sample_speeding_illegal.csv --task speeding --model cnn
python predict_sample.py --input sample_sidewalk_illegal.csv --task sidewalk --model cnn

# 3. Run the Quick/Accurate Dual-Mode Streaming Detector
python dualmode.py --task sidewalk --backend torch
```

---

## 📦 Project Structure

```text
SecureRide_Final_Project/
├── data/
│   ├── trips.csv                  # 36 hours of 5 Hz simulated telemetry (V, I, C, speed, surface)
│   ├── trips_meta.csv             # Trip metadata (rider mass, temperature, initial SoC, pack resistance)
│   ├── secureride_windows.npz     # Precomputed balanced sliding windows (L = 5..30)
│   ├── norm_stats.json            # Z-score normalization statistics (computed on train split only)
│   └── window_counts.csv          # Window distribution counts across splits
├── models/                        # Saved model checkpoints (.pt and .joblib)
│   ├── speeding_L10_HL2.pt        # ResNet-1D CNN (Speeding, SL=10, HL=2)
│   ├── sidewalk_L10_HL2.pt        # ResNet-1D CNN (Sidewalk, SL=10, HL=2)
│   ├── sidewalk_L20_HL2.pt        # ResNet-1D CNN (Accurate mode model)
│   ├── sidewalk_L5_HL1.pt         # ResNet-1D CNN (Quick mode model)
│   ├── speeding_L10_HL2_gru.pt    # Stacked GRU Baseline
│   └── sidewalk_L10_HL2.joblib    # Scikit-learn Baseline
├── results/                       # Experimental results, metrics, and figures
│   ├── metrics_torch.csv          # Full metrics for 1D CNN models
│   ├── metrics_torch_gru.csv      # Full metrics for GRU baseline
│   ├── metrics_sklearn.csv        # Full metrics for Scikit-learn baseline
│   ├── dualmode_sidewalk_torch.csv# Dual-mode 9-case evaluation metrics
│   ├── dualmode_speeding_torch.csv# Dual-mode speeding evaluation metrics
│   ├── calibration_summary.txt    # Physical calibration numbers
│   ├── fig2a_speed_voltage.png    # Speed vs terminal voltage drop (Fig 2a)
│   ├── fig2b_sidewalk_dv.png      # Road vs Sidewalk delta V spread (Fig 2b)
│   ├── fig5_accuracy_torch.png    # CNN Accuracy vs Sequence Length (Fig 5)
│   ├── fig9_dualmode_sidewalk_torch.png # Dual-mode 9-case evaluation plot (Fig 9)
│   └── fig_model_comparison.png   # CNN vs GRU vs Sklearn comparison
├── sample_speeding_illegal.csv    # Sample trace: sustained speeding (> 20 km/h)
├── sample_speeding_legal.csv      # Sample trace: legal cruising speed
├── sample_sidewalk_illegal.csv    # Sample trace: riding over segmented paving blocks
├── sample_sidewalk_legal.csv      # Sample trace: riding over smooth bicycle road
├── sample_mixed_trip.csv          # Sample trace: mixed trip with transitions
├── secureride_model.py            # Model definitions (PyTorch ResNet-1D, GRU, Sklearn)
├── secureride_data.py             # Data loader, window generation, augmentation
├── gen_dataset.py                 # Physics-based synthetic ride/battery simulator
├── calibration_plots.py           # Sanity-check calibration plots
├── train.py                       # Training sweep and evaluation pipeline
├── dualmode.py                    # Algorithm 1: Quick/Accurate dual-mode detector
├── predict_sample.py              # CLI for inference on sample telemetry CSV files
├── test_secureride.py             # 12-test automated unit test suite
├── requirements.txt               # Dependency specification
├── EXPERIMENT_LOG.md              # Detailed experimental log of executed commands
└── ACADEMIC_REPORT.md             # Full 17-section academic research report
```

---

## 🛠️ Installation & Environment Setup

### Prerequisites
- Python 3.10 or 3.11 (Python 3.11 recommended)
- Windows PowerShell, Linux Bash, or macOS Terminal

### Setup with Virtual Environment
```powershell
# Clone or navigate to the project directory
cd SecureRide_Final_Project

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔬 Experimental Workflows

### 1. Battery & Surface Physical Calibration (Figure 2)
Verify the physical battery principles that underpin SecureRide:
```powershell
python calibration_plots.py --root data --out results
```
- **Output:** Generates `results/fig2a_speed_voltage.png`, `results/fig2b_sidewalk_dv.png`, and `results/calibration_summary.txt`.
- **Finding:** Sidewalk riding exhibits **2.33x** greater voltage jitter variance ($\Delta V$) than smooth asphalt road.

### 2. Model Training Sweeps

#### Train ResNet-based 1D CNN with Channel Attention (Paper Architecture)
```powershell
python train.py --backend torch --arch cnn --tasks speeding sidewalk --lengths 5 10 20 --hls 1 2 --max-train 4000 --max-eval 3000 --epochs 15
```

#### Train GRU Baseline (Paper Sec. 6.1)
```powershell
python train.py --backend torch --arch gru --tasks speeding sidewalk --lengths 5 10 20 --hls 1 2 --max-train 4000 --max-eval 3000 --epochs 15
```

#### Train Fast Scikit-Learn Baseline
```powershell
python train.py --backend sklearn --tasks speeding sidewalk --lengths 5 10 20 30 --hls 1 2 3 --max-train 15000 --max-eval 10000
```

### 3. Quick/Accurate Dual-Mode Streaming Detection (Algorithm 1)
Evaluates 9 operating scenarios across time constraints ($\tau \in \{1.0, 1.5, 2.0\}\ \text{s}$) and accuracy targets ($\alpha$):
```powershell
# Dual-mode for sidewalk riding
python dualmode.py --task sidewalk --backend torch

# Dual-mode for speeding violation
python dualmode.py --task speeding --backend torch
```

---

## 🎯 Sample-Data Prediction CLI (`predict_sample.py`)

A production-ready command-line tool that loads saved checkpoints, validates input telemetry, normalizes features, and classifies illegal behavior with confidence scores.

### Speeding Detection Examples
```powershell
# Predict speeding on an illegal speeding trace
python predict_sample.py --input sample_speeding_illegal.csv --task speeding --model models/speeding_L10_HL2.pt

# Predict speeding on a legal driving trace
python predict_sample.py --input sample_speeding_legal.csv --task speeding --model models/speeding_L10_HL2.pt
```

### Sidewalk Riding Detection Examples
```powershell
# Predict sidewalk riding on paving blocks
python predict_sample.py --input sample_sidewalk_illegal.csv --task sidewalk --model models/sidewalk_L10_HL2.pt

# Predict sidewalk riding on bicycle asphalt road
python predict_sample.py --input sample_sidewalk_legal.csv --task sidewalk --model models/sidewalk_L10_HL2.pt
```

### Keyword Model Resolution
You can also use model keywords like `--model cnn`, `--model gru`, or `--model sklearn`:
```powershell
python predict_sample.py --input sample_sidewalk_illegal.csv --task sidewalk --model cnn
python predict_sample.py --input sample_sidewalk_illegal.csv --task sidewalk --model sklearn
```

### Expected Output Example
```text
======================================================================
  SecureRide: Illegal E-Scooter Behavior Predictor
======================================================================
Loaded input file: sample_speeding_illegal.csv (75 timesteps, 15.0 seconds)
Loading trained model checkpoint: models/speeding_L10_HL2.pt
Model Configuration: Task=speeding | Backend=torch | Arch=cnn | Window L=10 (2.0s) | HL=2
----------------------------------------------------------------------
PREDICTION RESULTS SUMMARY (SPEEDING DETECTION):
  Overall Trip Verdict      : >>> SPEEDING VIOLATION <<<
  Illegal Window Fraction   : 66/66 (100.0%)
  Average Violation Conf.   : 95.4%
  Inference Latency per Win : 3.04 ms
----------------------------------------------------------------------
Timeline Preview (First 8 detection slots):
 time_s  voltage_V  current_A prediction  confidence  prob_illegal
    2.0      37.97       4.65    ILLEGAL        96.1          96.1
    2.2      38.01       4.03    ILLEGAL        94.4          94.4
    2.4      37.83       5.24    ILLEGAL        93.7          93.7
    2.6      37.68       5.94    ILLEGAL        93.4          93.4
...
```

---

## 🧪 Automated Test Suite

Run the full test suite verifying dataset loading, normalization, feature ordering, physical augmentation laws, trip separation, model serialization, dual-mode transitions, and input validation:
```powershell
python -m unittest test_secureride.py
```
**Status:** 12/12 Tests Passing.

---

## 📊 Summary of Evaluated Results

| Model Architecture | Task | Sequence Length | Test Accuracy | F1-Score | Inference Latency | Model Footprint |
|---|---|---|---|---|---|---|
| **ResNet-1D CNN (HL2)** | Sidewalk Riding | SL 20 (4.0s) | **87.83%** | **0.8742** | 0.96 ms | 63.3 KB |
| **ResNet-1D CNN (HL2)** | Speeding Violation | SL 20 (4.0s) | **83.43%** | **0.8343** | 3.44 ms | 63.3 KB |
| **Stacked GRU (HL2)** | Sidewalk Riding | SL 20 (4.0s) | 84.33% | 0.8273 | 1.61 ms | 43.1 KB |
| **Stacked GRU (HL2)** | Speeding Violation | SL 20 (4.0s) | 82.67% | 0.8263 | 4.63 ms | 43.1 KB |
| **HistGradBoost (HL1)**| Sidewalk Riding | SL 30 (6.0s) | 87.77% | 0.8715 | 2.44 ms | 98.9 KB |
| **Dual-Mode System** | Sidewalk Riding | Adaptive | **93.8% Slots**| — | **0.57 s Detection**| Combined |

---

## ⚖️ Research Integrity & Disclaimer

The original experimental dataset gathered in Sweden and Korea is proprietary and was unavailable. All evaluations in this repository were conducted using the provided physics-inspired simulation dataset. Synthetic results are clearly distinguished from empirical real-world results. For complete details, see [`ACADEMIC_REPORT.md`](ACADEMIC_REPORT.md) and [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).


## 4. Added Experiments: Class Imbalance and Temporal Smoothing

This project includes two additional experiments beyond the base speeding/sidewalk classifiers.

### A. Class-imbalance robustness (`imbalance.py`)

The experiment changes the fraction of illegal windows in the training/evaluation setup and compares unweighted training with class-balanced weighting. It reports precision, recall, F1, false-positive rate, balanced accuracy and related metrics. Prevalence is the fraction of positive/illegal examples; it is not a claim about real-world violation prevalence.

```bash
python imbalance.py --backend torch --task sidewalk --L 10 --hl 2 \
  --prevalences 0.5 0.05 0.01 --weights none balanced --seeds 3
python imbalance.py --backend torch --task speeding --L 10 --hl 2 \
  --prevalences 0.5 0.05 0.01 --weights none balanced --seeds 3
```

Outputs are written to `results/imbalance_*.csv`, summary CSVs, and figures. For a quick smoke test, use fewer seeds or the `sklearn` backend first.

### B. Temporal smoothing (`smoothing.py`)

This experiment applies a majority-vote window to consecutive model predictions. `N=1` is the unsmoothed baseline. Larger `N` can suppress isolated false alarms, but may delay detection or suppress short genuine violations.

```bash
python smoothing.py --backend torch --task sidewalk --ns 1 3 5 7
python smoothing.py --backend torch --task speeding --ns 1 3 5 7
```

The script writes `results/smoothing_*.csv` and figures. Compare false-alarm events/rates and detection delay alongside precision, recall and F1; do not select a window using accuracy alone.

### Reproducibility notes

- Keep the same held-out trip split when comparing configurations.
- Select settings using validation data; use the test split only for final reporting.
- Record the exact command, seed, dependency versions and output filenames for every run.
- The dataset and sample traces in this repository are synthetic. These experiments measure performance under simulator assumptions and do not establish real-world scooter performance.
