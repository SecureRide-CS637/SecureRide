#!/usr/bin/env python3
"""
SecureRide Sample Prediction CLI (Kim et al., ACM TECS 2025)

Loads a trained SecureRide predictor model (PyTorch, Scikit-learn, or Keras)
and executes safety-threatening behavior detection on new scooter telemetry CSV data.

Supported Tasks:
  - 'speeding' : Detects excessive vehicle speed (> 20 km/h) via battery load drop.
  - 'sidewalk' : Detects illegal sidewalk riding via surface vibration voltage jitter.

Features required in input CSV:
  - 'voltage'  : Battery terminal voltage in Volts (V), typical range 32V - 42V.
  - 'current'  : Motor discharge / regen current in Amperes (A), typical range -5A to 25A.
  - 'capacity' : Remaining battery charge in milliampere-hours (mAh), typical range 0 - 8000 mAh.
Optional columns (ignored during inference): 't', 'speed', 'surface', 'trip_id'.

Usage Examples:
  python predict_sample.py --input sample_speeding_illegal.csv --task speeding --model models/speeding_L10_HL2.pt
  python predict_sample.py --input sample_sidewalk_illegal.csv --task sidewalk --model models/sidewalk_L10_HL2.pt
  python predict_sample.py --input sample_speeding_legal.csv --task speeding --model models/speeding_L10_HL1.joblib
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from secureride_model import load_predictor, SklearnPredictor, TorchPredictor

DT = 0.2  # 5 Hz sampling interval (0.2 s per slot)


def validate_input_data(df, min_length=5):
    """Rigorous schema, physical units, missing values, and length verification."""
    required_cols = ["voltage", "current", "capacity"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input CSV is missing required battery telemetry columns: {missing_cols}. "
            f"Expected columns: {required_cols}"
        )

    # Check for NaN / null values
    nan_counts = df[required_cols].isna().sum()
    if nan_counts.any():
        bad = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(f"Input CSV contains missing or NaN values in battery columns: {bad}")

    # Check sequence length
    n_rows = len(df)
    if n_rows < min_length:
        raise ValueError(
            f"Input sequence length ({n_rows} timesteps, {n_rows*DT:.1f}s) is shorter than "
            f"the required model window length (L={min_length} timesteps, {min_length*DT:.1f}s)."
        )

    # Physical unit sanity checks
    v_min, v_max = df["voltage"].min(), df["voltage"].max()
    if v_min < 25.0 or v_max > 55.0:
        print(f"[WARNING] Voltage values ({v_min:.1f}V to {v_max:.1f}V) outside expected 10S Li-ion pack range (30V-45V).")

    i_min, i_max = df["current"].min(), df["current"].max()
    if i_min < -30.0 or i_max > 45.0:
        print(f"[WARNING] Current values ({i_min:.1f}A to {i_max:.1f}A) outside normal e-scooter range (-10A to 30A).")

    c_min, c_max = df["capacity"].min(), df["capacity"].max()
    if c_min < 0 or c_max > 25000:
        print(f"[WARNING] Capacity values ({c_min:.0f} to {c_max:.0f} mAh) outside expected range (0 to 15000 mAh).")


def resolve_model_path(models_dir, task, model_arg, backend=None):
    """Resolves model path from user input (can be full path or short name like 'cnn' or 'sklearn')."""
    if os.path.exists(model_arg):
        return model_arg

    # Check if inside models_dir
    direct_in_dir = os.path.join(models_dir, model_arg)
    for ext in ("", ".pt", ".joblib", ".keras"):
        p = direct_in_dir + ext
        if os.path.exists(p):
            return p

    # Heuristic resolution (e.g. --model cnn or --model sklearn or --model gru)
    arch_key = model_arg.lower()
    candidates = []
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.startswith(task) and (
                (arch_key in ("cnn", "torch") and f.endswith(".pt") and "gru" not in f) or
                (arch_key == "gru" and f.endswith(".pt") and "gru" in f) or
                (arch_key == "sklearn" and f.endswith(".joblib")) or
                (arch_key == "keras" and f.endswith(".keras"))
            ):
                candidates.append(os.path.join(models_dir, f))

    if candidates:
        # Prefer moderate sequence length like L10 or L15
        candidates.sort(key=lambda x: (0 if "L10" in x else 1, x))
        return candidates[0]

    raise FileNotFoundError(
        f"Could not resolve model '{model_arg}' for task '{task}' in '{models_dir}'. "
        f"Please provide an exact path, e.g. 'models/{task}_L10_HL2.pt'."
    )


def main():
    ap = argparse.ArgumentParser(description="SecureRide Sample-Data Inference CLI")
    ap.add_argument("--input", required=True, help="Path to input CSV containing battery measurements")
    ap.add_argument("--task", choices=["speeding", "sidewalk"], required=True, help="Target safety threat")
    ap.add_argument("--model", default="cnn", help="Path to trained model checkpoint or model keyword (cnn, gru, sklearn)")
    ap.add_argument("--models-dir", default="models", help="Directory containing saved model files")
    ap.add_argument("--stats", default="data/norm_stats.json", help="Path to normalization statistics JSON")
    ap.add_argument("--output", default=None, help="Optional output path to export window-level predictions to CSV")
    args = ap.parse_args()

    print("=" * 70)
    print("  SecureRide: Illegal E-Scooter Behavior Predictor")
    print("=" * 70)

    # 1. Load telemetry CSV
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded input file: {args.input} ({len(df)} timesteps, {len(df)*DT:.1f} seconds)")

    # 2. Resolve and load model
    model_path = resolve_model_path(args.models_dir, args.task, args.model)
    print(f"Loading trained model checkpoint: {model_path}")
    predictor = load_predictor(model_path)
    L = getattr(predictor, "L", 10)
    hl = getattr(predictor, "hl", 1)
    arch = getattr(predictor, "arch", "unknown")
    backend = getattr(predictor, "backend", "unknown")

    print(f"Model Configuration: Task={args.task} | Backend={backend} | Arch={arch} | Window L={L} ({L*DT:.1f}s) | HL={hl}")

    # 3. Validate input against model window size
    validate_input_data(df, min_length=L)

    # 4. Normalization
    if not os.path.exists(args.stats):
        raise FileNotFoundError(f"Normalization statistics file not found at {args.stats}")

    stats = json.load(open(args.stats))
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    raw_features = df[["voltage", "current", "capacity"]].to_numpy(dtype=np.float32)
    norm_features = (raw_features - mean) / std

    # 5. Build sliding windows (step = 1 slot / 0.2s)
    W = np.lib.stride_tricks.sliding_window_view(norm_features, L, axis=0).transpose(0, 2, 1) # (N_windows, L, 3)
    W = np.ascontiguousarray(W, dtype=np.float32)

    # 6. Inference
    preds = predictor.predict(W)
    probs = predictor.predict_proba(W)

    # Timestamps correspond to window end
    window_times = [round((i + L) * DT, 2) for i in range(len(preds))]

    # 7. Overall summary metrics
    total_windows = len(preds)
    illegal_windows = int((preds == 1).sum())
    illegal_pct = 100.0 * illegal_windows / total_windows
    mean_conf_illegal = float(probs[:, 1].mean() * 100.0)

    task_label_illegal = "SPEEDING VIOLATION" if args.task == "speeding" else "SIDEWALK RIDING"
    overall_verdict = task_label_illegal if illegal_pct >= 50.0 else "LEGAL DRIVING"

    print("-" * 70)
    print(f"PREDICTION RESULTS SUMMARY ({args.task.upper()} DETECTION):")
    print(f"  Overall Trip Verdict      : >>> {overall_verdict} <<<")
    print(f"  Illegal Window Fraction   : {illegal_windows}/{total_windows} ({illegal_pct:.1f}%)")
    print(f"  Average Violation Conf.   : {mean_conf_illegal:.1f}%")
    print(f"  Inference Latency per Win : {predictor.exec_ms(50):.2f} ms")
    print("-" * 70)

    # 8. Print sample timeline preview (first 10 and last 5 windows)
    res_df = pd.DataFrame({
        "time_s": window_times,
        "voltage_V": [round(float(df["voltage"].iloc[i + L - 1]), 2) for i in range(len(preds))],
        "current_A": [round(float(df["current"].iloc[i + L - 1]), 2) for i in range(len(preds))],
        "prediction": ["ILLEGAL" if p == 1 else "LEGAL" for p in preds],
        "confidence": [round(float(probs[i, p] * 100.0), 1) for i, p in enumerate(preds)],
        "prob_illegal": [round(float(probs[i, 1] * 100.0), 1) for i in range(len(preds))]
    })

    print("Timeline Preview (First 8 detection slots):")
    print(res_df.head(8).to_string(index=False))

    if len(res_df) > 12:
        print("  ...")
        print("Timeline Preview (Last 4 detection slots):")
        print(res_df.tail(4).to_string(index=False))

    # 9. Export if requested
    if args.output:
        res_df.to_csv(args.output, index=False)
        print(f"\nDetailed per-slot predictions exported to: {args.output}")

    print("=" * 70)


if __name__ == "__main__":
    main()
