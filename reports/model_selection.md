# SecureRide Heterogeneous Model Selection Report (Corrected)

> Selected model pair matching paper Eq. 5 & Eq. 6 (J. Kim et al., *SecureRide*, ACM TECS 2025).

## 1. Selected Model Pair

| Attribute | Quick Model (θ_quick) | Accurate Model (θ_accurate) |
|---|---|---|
| **Sequence Length ($L_{seq}$)** | **5 seconds** | **20 seconds** |
| **Residual Depth ($HL$)** | **HL1** | **HL3** |
| **Parameters** | **10,762** | **30,154** (2.8× complexity) |
| **TFLite Size** | **28.4 KB** | **65.0 KB** |
| **Validation Accuracy** | 95.54% | 98.03% |
| **Test Accuracy** | 95.68% | 97.04% |
| **Test Precision** | 99.43% | 98.95% |
| **Test Recall** | 95.27% | 97.19% |
| **Balanced Accuracy** | 96.41% | 96.86% |
| **Buffering Latency ($T_{buffer}$)** | 5.0 s | 20.0 s |
| **Execution Latency ($T_{exec}$)** | 10.0 ms | 400.0 ms |
| **Model Artifacts** | `theta_quick.h5`, `.tflite` | `theta_accurate.h5`, `.tflite` |

## 2. Latency & Sampling Constraint Note (§A.3)
> **Important Physical Latency Caveat**: In our 1 Hz Xiaomi BLE dataset, filling a sequence window takes `L` seconds (`T_buffer = L / 1.0 Hz`). Thus, detection latency is strictly bound by `L × 1 s` (5 s for θ_quick, 20 s for θ_accurate). The paper's headline 1.01 s total detection latency was achieved on hardware with ~3+ Hz PMIC high-rate telemetry and cannot be physically matched at 1 Hz without down-sampling or extrapolation.

## 3. Dual-Mode Heterogeneous Architecture Alignment (§B)
- **Quick Model ($\theta_{quick}$, SL5 HL1)**: Operates continuously during normal operation. Consumes minimal memory (28.4 KB) and tiny execution latency (~10 ms), evaluating short 5 s windows to rapidly catch speeding transitions.
- **Accurate Model ($\theta_{accurate}$, SL20 HL3)**: Genuinely more complex than the quick model (3 residual blocks, 30,154 parameters vs 10,762). Triggered when the quick model trips to confirm speeding across a wide 20 s horizon, achieving 98.03% validation accuracy and 97.04% test accuracy with 98.95% precision.
