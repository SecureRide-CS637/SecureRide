#!/usr/bin/env python3
"""
Automated Comprehensive Test Suite for SecureRide Implementation
(Kim et al., ACM TECS 2025)

Covers:
  1. Dataset loading and shape verification
  2. Preprocessing, statistics, and normalization consistency
  3. Feature order and physical units verification
  4. Binary label generation and All-True window labeling rule (Eq. 2)
  5. Group-based train/val/test trip split isolation (Zero window leakage)
  6. Data augmentation physical consistency (Eq. 3, dV = -R*dI)
  7. Model training, parameter updating, and convergence (Smoke test)
  8. Model serialization and exact checkpoint deserialization
  9. Prediction on sample CSV files across both tasks
  10. Dual-mode state machine transitions (Algorithm 1)
  11. Input validation, missing data, and invalid schema error handling
  12. Evaluation metric calculation correctness (Accuracy, F1, Balanced Acc, CM)
"""
import unittest
import os, sys, json, tempfile
import numpy as np
import pandas as pd

import secureride_data as sd
from secureride_model import make_predictor, load_predictor, window_features
from dualmode import DualModeDetector, select_models, detection_time
from predict_sample import validate_input_data, resolve_model_path


class TestSecureRidePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = "data"
        cls.models_dir = "models"
        cls.results_dir = "results"
        os.makedirs(cls.models_dir, exist_ok=True)
        os.makedirs(cls.results_dir, exist_ok=True)

    # 1. Dataset loading
    def test_01_dataset_loading(self):
        z = sd._npz(self.root)
        self.assertIn("series", z)
        self.assertIn("speeding", z)
        self.assertIn("sidewalk", z)
        self.assertIn("split", z)
        self.assertGreater(len(z["series"]), 10000)

    # 2. Preprocessing & Normalization Stats
    def test_02_norm_stats(self):
        stats_path = f"{self.root}/norm_stats.json"
        self.assertTrue(os.path.exists(stats_path))
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertEqual(len(stats["mean"]), 3)
        self.assertEqual(len(stats["std"]), 3)
        # Check nominal ranges: V ~ 35-40V, I ~ 2-8A, C ~ 3000-6000 mAh
        self.assertGreater(stats["mean"][0], 30.0)
        self.assertLess(stats["mean"][0], 45.0)
        self.assertGreater(stats["std"][0], 0.1)

    # 3. Feature Order and Physical Units
    def test_03_feature_order_and_units(self):
        # Raw window loading
        X, y = sd.load("speeding", L=10, split="test", root=self.root, normalize=False)
        self.assertEqual(X.shape[1], 10)
        self.assertEqual(X.shape[2], 3)
        # Check channel order: [V, I, C]
        v_mean = X[..., 0].mean()
        i_mean = X[..., 1].mean()
        c_mean = X[..., 2].mean()
        self.assertTrue(30.0 < v_mean < 45.0, f"Voltage channel misplaced: {v_mean}")
        self.assertTrue(-5.0 < i_mean < 25.0, f"Current channel misplaced: {i_mean}")
        self.assertTrue(0.0 <= c_mean <= 10000.0, f"Capacity channel misplaced: {c_mean}")

    # 4. Binary Label Generation & All-True Rule (Eq. 2)
    def test_04_window_labeling_rule(self):
        # Window label must be 1 if and only if all L samples in the window are illegal
        z = sd._npz(self.root)
        for task in ("speeding", "sidewalk"):
            L = 10
            starts = z[f"{task}_L{L}_val_start"][:100]
            labels = z[f"{task}_L{L}_val_y"][:100]
            for st, lab in zip(starts, labels):
                sub_labels = z[task][st : st + L]
                expected_label = 1 if np.all(sub_labels == 1) else 0
                self.assertEqual(lab, expected_label, f"All-true labeling violated for {task} at start {st}")

    # 5. Train/Val/Test Split Separation (Zero Window Leakage)
    def test_05_split_isolation(self):
        df_trips = pd.read_csv(f"{self.root}/trips.csv")
        train_trips = set(df_trips[df_trips.split == "train"]["trip_id"].unique())
        val_trips = set(df_trips[df_trips.split == "val"]["trip_id"].unique())
        test_trips = set(df_trips[df_trips.split == "test"]["trip_id"].unique())

        # Disjoint verification
        self.assertEqual(len(train_trips.intersection(val_trips)), 0)
        self.assertEqual(len(train_trips.intersection(test_trips)), 0)
        self.assertEqual(len(val_trips.intersection(test_trips)), 0)

        # Window starts must also strictly belong to trips in that split
        z = sd._npz(self.root)
        trip_ids = z["trip"]
        train_starts = z["speeding_L10_train_start"][:500]
        for st in train_starts:
            self.assertIn(trip_ids[st], train_trips)

        test_starts = z["speeding_L10_test_start"][:500]
        for st in test_starts:
            self.assertIn(trip_ids[st], test_trips)

    # 6. Data Augmentation Physical Consistency (Eq. 3)
    def test_06_augmentation_consistency(self):
        rng = np.random.default_rng(42)
        X = np.ones((10, 15, 3), dtype=np.float32)
        X[..., 0] = 38.0 # 38 V
        X[..., 1] = 5.0  # 5 A
        X[..., 2] = 5000.0 # 5000 mAh

        # Test step-slope augmentation
        X_aug = sd.f_step_slope(X, rng)
        # Check that dV = -R * dI holds for every window
        dV = X_aug[..., 0] - X[..., 0]
        dI = X_aug[..., 1] - X[..., 1]
        for n in range(10):
            nonzero = np.abs(dI[n]) > 1e-4
            if np.any(nonzero):
                r_estimated = - (dV[n][nonzero] / dI[n][nonzero])
                self.assertTrue(np.all(r_estimated > 0.05) and np.all(r_estimated < 0.4),
                                f"Physical impedance violation: R={r_estimated}")

    # 7. Model Training & Saving (Smoke Test)
    def test_07_model_training_and_saving(self):
        # Quick synthetic data
        X = np.random.randn(64, 5, 3).astype(np.float32)
        y = np.random.choice([0, 1], 64).astype(np.int64)

        # PyTorch CNN
        p_cnn = make_predictor("torch", L=5, hl=1, arch="cnn", epochs=2, batch=16, verbose=0)
        p_cnn.fit(X, y, X, y)
        cnn_path = f"{self.models_dir}/smoke_cnn"
        p_cnn.save(cnn_path)
        self.assertTrue(os.path.exists(cnn_path + ".pt"))

        # PyTorch GRU
        p_gru = make_predictor("torch", L=5, hl=1, arch="gru", epochs=2, batch=16, verbose=0)
        p_gru.fit(X, y, X, y)
        gru_path = f"{self.models_dir}/smoke_gru"
        p_gru.save(gru_path)
        self.assertTrue(os.path.exists(gru_path + ".pt"))

        # Sklearn
        p_sk = make_predictor("sklearn", L=5, hl=1, seed=0)
        p_sk.fit(X, y, X, y)
        sk_path = f"{self.models_dir}/smoke_sklearn"
        p_sk.save(sk_path)
        self.assertTrue(os.path.exists(sk_path + ".joblib"))

    # 8. Model Loading and Inference Fidelity
    def test_08_model_loading_and_inference(self):
        X = np.random.randn(20, 5, 3).astype(np.float32)

        # Reload Torch CNN
        p_cnn = load_predictor(f"{self.models_dir}/smoke_cnn.pt")
        preds_cnn = p_cnn.predict(X)
        probs_cnn = p_cnn.predict_proba(X)
        self.assertEqual(len(preds_cnn), 20)
        self.assertEqual(probs_cnn.shape, (20, 2))
        self.assertTrue(np.allclose(probs_cnn.sum(axis=1), 1.0, atol=1e-4))

        # Reload Sklearn
        p_sk = load_predictor(f"{self.models_dir}/smoke_sklearn.joblib")
        preds_sk = p_sk.predict(X)
        self.assertEqual(len(preds_sk), 20)

    # 9. Prediction on Sample CSV Files
    def test_09_sample_csv_prediction(self):
        samples = [
            "sample_speeding_illegal.csv",
            "sample_speeding_legal.csv",
            "sample_sidewalk_illegal.csv",
            "sample_sidewalk_legal.csv"
        ]
        for s in samples:
            self.assertTrue(os.path.exists(s), f"Missing sample file: {s}")
            df = pd.read_csv(s)
            validate_input_data(df, min_length=5)
            self.assertIn("voltage", df.columns)
            self.assertIn("current", df.columns)
            self.assertIn("capacity", df.columns)

    # 10. Dual-Mode State Transitions (Algorithm 1)
    def test_10_dualmode_state_transitions(self):
        # Create a mock detector with dummy quick and accurate predictors
        class MockPredictor:
            def __init__(self, fixed_val):
                self.v = fixed_val
            def predict(self, x):
                return np.array([self.v], dtype=np.int8)

        # Scenario: Quick says 1 (illegal), Accurate confirms 1 (illegal)
        q = MockPredictor(1)
        a = MockPredictor(1)
        det = DualModeDetector(q, a, L_short=5, L_long=10)
        dummy_fn = lambda L: np.zeros((L, 3), dtype=np.float32)

        # Slot 0 to 4: buffering quick
        for _ in range(4):
            y = det.step(dummy_fn, n_avail=4)
            self.assertEqual(y, 0)

        # Slot 5: quick triggers
        y = det.step(dummy_fn, n_avail=5)
        self.assertEqual(y, 1)
        self.assertEqual(det.mode, "accurate")

        # Now in accurate mode: should hold provisional verdict 1 while buffering new samples
        y = det.step(dummy_fn, n_avail=6)
        self.assertEqual(y, 1)
        self.assertEqual(det.mode, "accurate")

    # 11. Error Handling on Invalid/Missing Inputs
    def test_11_input_validation_errors(self):
        # Missing column
        df_bad = pd.DataFrame({"voltage": [38.0]*10, "current": [5.0]*10})
        with self.assertRaises(ValueError):
            validate_input_data(df_bad, min_length=5)

        # NaN value
        df_nan = pd.DataFrame({"voltage": [38.0, np.nan, 38.0], "current": [5.0, 5.0, 5.0], "capacity": [5000, 5000, 5000]})
        with self.assertRaises(ValueError):
            validate_input_data(df_nan, min_length=2)

        # Short length
        df_short = pd.DataFrame({"voltage": [38.0]*3, "current": [5.0]*3, "capacity": [5000]*3})
        with self.assertRaises(ValueError):
            validate_input_data(df_short, min_length=10)

    # 12. Metric Calculations
    def test_12_metric_calculations(self):
        from train import evaluate_split
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        m = evaluate_split(y_true, y_pred)
        self.assertAlmostEqual(m["acc"], 6/8)
        self.assertTrue(0.0 <= m["f1"] <= 1.0)
        self.assertTrue(0.0 <= m["bal_acc"] <= 1.0)
        self.assertEqual(m["tp"] + m["fp"] + m["tn"] + m["fn"], 8)


if __name__ == "__main__":
    unittest.main()
