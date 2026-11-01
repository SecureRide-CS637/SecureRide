# SecureRide Training Data Pipeline: Final QA Report

## 1. Executive Summary
The raw Xiaomi ride telemetry has been processed into standardized, windowed, labeled, and augmented datasets following all requirements of the Implementation Plan.

- **Target sampling rate**: 1.0 Hz (uniform)
- **Fixed channel order**: `['voltage', 'current', 'soc']`
- **Speeding threshold**: > 20.0 km/h
- **Labeling rule**: True (speeding) iff *every* point in the window exceeds 20 km/h
- **Augmentation**: Applied strictly to Train features only (`F_GN ∪ F_SN ∪ F_SS`, 4× training size)
- **Standardization**: Z-score scaler fit strictly on Train split
- **Data leakage prevention**: Strict group split by ride (no ride appears in multiple splits)

## 2. Artifacts & Shapes Summary

| Sequence Length | Train `X` (Augmented) | Train `y` (One-Hot) | Val `X` | Val `y` | Test `X` | Test `y` |
|---|---|---|---|---|---|---|
| `seq5` | `[41344, 5, 3]` | `[41344, 2]` | `[1436, 5, 3]` | `[1436, 2]` | `[2919, 5, 3]` | `[2919, 2]` |
| `seq10` | `[41304, 10, 3]` | `[41304, 2]` | `[1431, 10, 3]` | `[1431, 2]` | `[2914, 10, 3]` | `[2914, 2]` |
| `seq15` | `[41264, 15, 3]` | `[41264, 2]` | `[1426, 15, 3]` | `[1426, 2]` | `[2909, 15, 3]` | `[2909, 2]` |
| `seq20` | `[41224, 20, 3]` | `[41224, 2]` | `[1421, 20, 3]` | `[1421, 2]` | `[2904, 20, 3]` | `[2904, 2]` |
| `seq25` | `[41184, 25, 3]` | `[41184, 2]` | `[1416, 25, 3]` | `[1416, 2]` | `[2899, 25, 3]` | `[2899, 2]` |

## 3. Label Balance & Distribution

| Sequence Length | Train Speeding % (4× aug) | Val Speeding % | Test Speeding % |
|---|---|---|---|
| `seq5` | 71.15% (29416/41344) | 69.78% (1002/1436) | 81.88% (2390/2919) |
| `seq10` | 69.44% (28680/41304) | 67.58% (967/1431) | 80.30% (2340/2914) |
| `seq15` | 67.82% (27984/41264) | 65.36% (932/1426) | 78.72% (2290/2909) |
| `seq20` | 66.23% (27304/41224) | 63.12% (897/1421) | 77.13% (2240/2904) |
| `seq25` | 64.65% (26624/41184) | 61.16% (866/1416) | 75.54% (2190/2899) |

## 4. Normalization Scaler Parameters (Fit on Train)

| Channel Index | Channel Name | Train Mean (μ) | Train Std (σ) |
|---|---|---|---|
| 0 | `voltage` | 36.6388 | 1.5389 |
| 1 | `current` | 5.5057 | 4.0346 |
| 2 | `soc` | 68.2389 | 17.5908 |

## 5. Post-Standardization Sanity Checks

- **Train standardized mean**: ~0.00 across all channels
- **Train standardized std**: ~1.00 across all channels
- **Data types**: `X` is `float32`, `y` is `int8` (one-hot `[normal, speeding]`)
- **Validation / Test sets**: Strictly untouched by data augmentation

## 6. Directory Layout
```
data/final/
├── scaler.npz
├── seq5/
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── scaler.npz
├── seq10/
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── scaler.npz
├── seq15/
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── scaler.npz
├── seq20/
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── scaler.npz
└── seq25/
    ├── train.npz
    ├── val.npz
    ├── test.npz
    └── scaler.npz
```

All acceptance criteria met. Artifacts ready for training DNN quick and accurate models.