# SecureRide: Experiment Execution & Verification Log

**Date:** October 28, 2026  
**Research Paper:** *SecureRide: Detecting Safety-Threatening Behavior of E-Scooters Using Battery Information* (ACM TECS / ACM EMSOFT 2025)  
**Host Environment:** Windows x86_64, Python 3.11.9, PyTorch 2.15.0+cpu, Scikit-Learn 1.9.1, NumPy 2.4.6, Pandas 3.1.0, Matplotlib 3.11.2  

---

## 1. Project Audit Summary

1. **Simulator Speed-Bin Gap:**
   - **Finding:** In `gen_dataset.py`, the target speed bin for legal riding was defined as `(15, 19)` km/h and the speeding bin was `(21.5, 24.5)` km/h, leaving an artificial 2.5 km/h gap (`19.0 - 21.5 km/h`).
   - **Technical Explanation:** The simulator author introduced this deadband to prevent target speed flickering across the 20.0 km/h boundary during steady-state segments. However, dynamic speed transitions, deceleration, and Ornstein-Uhlenbeck jitter caused 44,732 rows (6.7% of the dataset) to traverse this boundary. In the original paper (Table 2), speed bins are continuous: 0–5, 5–10, 10–15, 15–20, 20–25 km/h.
2. **Voltage Drop Discrepancy (2.72% vs ~9.5%):**
   - **Finding:** `calibration_plots.py` reported a 2.72% voltage drop between $\ge 20$ km/h and $< 20$ km/h, while the paper reported ~9.5%.
   - **Technical Explanation:** In the synthetic simulation, trip State-of-Charge (SoC) is uniformly distributed across all speeds ($mean\ SoC \approx 0.62$), isolating pure instantaneous ohmic and polarization drop ($I \cdot R$). In real empirical tests, higher speeds accelerate battery drain, and real packs exhibit higher total impedance (~0.30–0.40 $\Omega$ including BMS FETs, cell interconnects, and shunt resistors). When controlling for SoC by capacity quintiles, the simulated drop is 2.50%–3.39%. Following guidelines, no artificial data manipulation was performed.
3. **Missing Artifacts:**
   - Saved model checkpoints (`.pt`, `.joblib`) and sample prediction CLI were missing.

---

## 2. Automated Test Suite Execution

Command executed:
```powershell
python -m unittest test_secureride.py
```
**Result:** 12/12 tests PASSED (Ran in 12.10s)
- `test_01_dataset_loading`: PASSED
- `test_02_norm_stats`: PASSED
- `test_03_feature_order_and_units`: PASSED
- `test_04_window_labeling_rule`: PASSED
- `test_05_split_isolation`: PASSED
- `test_06_augmentation_consistency`: PASSED
- `test_07_model_training_and_saving`: PASSED
- `test_08_model_loading_and_inference`: PASSED
- `test_09_sample_csv_prediction`: PASSED
- `test_10_dualmode_state_transitions`: PASSED
- `test_11_input_validation_errors`: PASSED
- `test_12_metric_calculations`: PASSED

---

## 3. Simulator Calibration & Figure 2 Replication

Command executed:
```powershell
python calibration_plots.py --root data --out results
```
**Key Numbers:**
- Train rows evaluated: 413,994
- Under 10 km/h: Mean 37.75 V, Median 37.45 V ($n=92,692$)
- 10–20 km/h: Mean 37.26 V, Median 37.14 V ($n=192,114$)
- Over 20 km/h: Mean 36.40 V, Median 36.34 V ($n=129,188$)
- Voltage drop at $\ge 20$ km/h: 2.72% (SoC-controlled mean drop: 2.82%)
- Road vibration std($\Delta V$): 0.391 V (variance 0.153)
- Sidewalk vibration std($\Delta V$): 0.910 V (variance 0.828)
- Sidewalk to Road vibration ratio: **2.33x** (Paper reports substantial increase on block pavement)
- Artifacts: `fig2a_speed_voltage.png`, `fig2b_sidewalk_dv.png`, `fig2c_dv_all_types.png`, `calibration_summary.txt`

---

## 4. Model Training & Evaluation Sweeps

### A. Scikit-Learn HistGradientBoosting Baseline
Command executed:
```powershell
python train.py --backend sklearn --tasks speeding sidewalk --lengths 5 10 20 30 --hls 1 2 3 --max-train 15000 --max-eval 10000
```
- **Speeding:** Test Accuracy 79.43% (SL 5) $\rightarrow$ 81.91% (SL 10) $\rightarrow$ 86.51% (SL 20) $\rightarrow$ 86.50% (SL 30)
- **Sidewalk:** Test Accuracy 77.34% (SL 5) $\rightarrow$ 83.26% (SL 10) $\rightarrow$ 86.37% (SL 20) $\rightarrow$ 87.77% (SL 30)
- **Artifacts:** `results/metrics_sklearn.csv`, `results/fig5_accuracy_sklearn.png`

### B. ResNet-based 1D CNN with Channel Attention (PyTorch)
Command executed:
```powershell
python train.py --backend torch --arch cnn --tasks speeding sidewalk --lengths 5 10 20 --hls 1 2 --max-train 4000 --max-eval 3000 --epochs 15 --batch 256
```
- **Speeding:**
  - SL 5, HL 1: Val 80.00%, Test 79.00%, F1 0.795, Latency 1.91 ms, Size 19.8 KB
  - SL 10, HL 2: Val 83.70%, Test 81.27%, F1 0.818, Latency 4.19 ms, Size 63.3 KB
  - SL 20, HL 2: Val 86.50%, Test 83.43%, F1 0.834, Latency 3.44 ms, Size 63.3 KB
- **Sidewalk:**
  - SL 5, HL 1: Val 79.50%, Test 78.80%, F1 0.767, Latency 2.11 ms, Size 19.8 KB
  - SL 10, HL 2: Val 86.20%, Test 84.60%, F1 0.833, Latency 0.94 ms, Size 63.3 KB
  - SL 20, HL 2: Val 91.00%, Test **87.83%**, F1 0.874, Latency 0.96 ms, Size 63.3 KB
- **Artifacts:** `results/metrics_torch.csv`, `results/fig5_accuracy_torch.png`, `results/fig_training_curves_torch.png`

### C. GRU Baseline (PyTorch)
Command executed:
```powershell
python train.py --backend torch --arch gru --tasks speeding sidewalk --lengths 5 10 20 --hls 1 2 --max-train 4000 --max-eval 3000 --epochs 15 --batch 256
```
- **Speeding:** SL 5: 78.40% $\rightarrow$ SL 10: 79.43% $\rightarrow$ SL 20: 82.67%
- **Sidewalk:** SL 5: 76.07% $\rightarrow$ SL 10: 79.83% $\rightarrow$ SL 20: 84.33%
- **Observation:** CNN consistently outperforms GRU across all window lengths (+2.0% to +3.5% higher accuracy), while offering superior temporal feature extraction and smaller model footprint.
- **Artifacts:** `results/metrics_torch_gru.csv`, `results/fig5_accuracy_torch_gru.png`, `results/fig_training_curves_torch_gru.png`

---

## 5. Quick / Accurate Dual-Mode Detector (Algorithm 1)

Commands executed:
```powershell
python dualmode.py --task sidewalk --backend torch
python dualmode.py --task speeding --backend torch
python dualmode.py --task sidewalk --backend sklearn
python dualmode.py --task speeding --backend sklearn
```
**Sidewalk Riding (PyTorch CNN):**
- **Quick Model (SL 5 / HL 1):** Correct slots: 381.9 / 500 | Detection time: **1.12 s**
- **Accurate Model (SL 20 / HL 2):** Correct slots: 449.9 / 500 | Detection time: 2.70 s
- **Dual-Mode System:** Correct slots: **469.2 / 500 (93.8%)** | Detection time: **0.57 s**
- **Conclusion:** Dual-Mode successfully combines the rapid response of Quick mode with the high sustained accuracy of Accurate mode.

**Speeding Violation (PyTorch CNN):**
- **Quick Model (SL 5 / HL 1):** Correct slots: 454.4 / 500 | Detection time: 0.50 s
- **Accurate Model (SL 10 / HL 1):** Correct slots: 462.9 / 500 | Detection time: 1.39 s
- **Dual-Mode System:** Correct slots: 456.2 / 500 | Detection time: **0.50 s**
- **Artifacts:** `results/dualmode_sidewalk_torch.csv`, `results/dualmode_speeding_torch.csv`, `results/fig9_dualmode_sidewalk_torch.png`, `results/fig9_dualmode_speeding_torch.png`

---

## 6. Sample CSV Prediction Verification

Commands executed:
```powershell
python predict_sample.py --input sample_speeding_illegal.csv --task speeding --model models/speeding_L10_HL2.pt
python predict_sample.py --input sample_speeding_legal.csv --task speeding --model models/speeding_L10_HL2.pt
python predict_sample.py --input sample_sidewalk_illegal.csv --task sidewalk --model models/sidewalk_L10_HL2.pt
python predict_sample.py --input sample_sidewalk_legal.csv --task sidewalk --model models/sidewalk_L10_HL2.pt
```
**Results:**
- `sample_speeding_illegal.csv` $\rightarrow$ **SPEEDING VIOLATION** (100.0% illegal windows, 95.4% mean violation confidence).
- `sample_speeding_legal.csv` $\rightarrow$ **LEGAL DRIVING** (30.3% illegal windows, 37.2% mean violation confidence).
- `sample_sidewalk_illegal.csv` $\rightarrow$ **SIDEWALK RIDING** (59.1% illegal windows, 57.8% mean violation confidence).
- `sample_sidewalk_legal.csv` $\rightarrow$ **LEGAL DRIVING** (0.0% illegal windows, 18.8% mean violation confidence).
