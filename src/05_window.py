import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG

import numpy as np
import pandas as pd

def window_label(point_label, L):
    """
    Paper §4.1.2: true (1) iff every point in the window is speeding.
    Uses rolling sum == L.
    Returns: array of shape (len(point_label) - L + 1,) dtype int8.
    """
    if len(point_label) < L:
        return np.empty(0, dtype=np.int8)
    s = np.convolve(point_label.astype(int), np.ones(L, dtype=int), mode="valid")
    return (s == L).astype(np.int8)


def make_windows(features, point_labels, L, step=1):
    """
    Produce sliding windows of shape (N, L, 3) and window labels (N,).
    Window is labeled 1 iff ALL points in the window are speeding.
    """
    if len(features) < L:
        return np.empty((0, L, features.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int8)

    # sliding_window_view gives shape (N, 3, L) -> swapaxes gives (N, L, 3)
    X = np.lib.stride_tricks.sliding_window_view(features, L, axis=0).swapaxes(1, 2)[::step]
    y = window_label(point_labels, L)[::step]
    return X.astype(np.float32), y.astype(np.int8)

def generate_windowed_datasets():
    processed_dir = CONFIG["paths"]["processed"]
    windowed_dir = CONFIG["paths"]["windowed"]
    reports_dir = CONFIG["paths"]["reports"]
    windowed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    ride_files = sorted(list(processed_dir.glob("ride_*.npz")))
    if not ride_files:
        raise FileNotFoundError(f"No processed ride files found in {processed_dir}. Run 04_label.py first.")

    step = CONFIG.get("window_step", 1)
    seq_lengths = CONFIG["seq_lengths"]
    print(f"Generating windowed datasets for seq_lengths: {seq_lengths} with step={step}")

    # Load all rides into memory
    rides_data = []
    for f in ride_files:
        d = np.load(f)
        ride_id = f.stem.replace("ride_", "")
        rides_data.append({
            "ride_id": ride_id,
            "features": d["features"],
            "point_label": d["point_label"],
            "t": d["t"],
            "speed": d["speed"],
        })

    window_reports = []

    for L in seq_lengths:
        all_X = []
        all_y = []
        all_ride_ids = []

        for r in rides_data:
            X_ride, y_ride = make_windows(r["features"], r["point_label"], L, step=step)
            n_win = len(X_ride)
            if n_win > 0:
                all_X.append(X_ride)
                all_y.append(y_ride)
                all_ride_ids.append(np.array([r["ride_id"]] * n_win, dtype=object))

        X_cat = np.concatenate(all_X, axis=0) if all_X else np.empty((0, L, 3), dtype=np.float32)
        y_cat = np.concatenate(all_y, axis=0) if all_y else np.empty(0, dtype=np.int8)
        ride_ids_cat = np.concatenate(all_ride_ids, axis=0) if all_ride_ids else np.empty(0, dtype=object)

        out_file = windowed_dir / f"seq{L}.npz"
        np.savez_compressed(
            out_file,
            X=X_cat,
            y=y_cat,
            ride_id=ride_ids_cat,
        )

        pos_count = int(y_cat.sum())
        total_count = len(y_cat)
        pos_ratio = (pos_count / total_count * 100) if total_count > 0 else 0.0

        stat = {
            "seq_len": L,
            "file": out_file.name,
            "N_windows": total_count,
            "X_shape": str(X_cat.shape),
            "y_shape": str(y_cat.shape),
            "speeding_windows": pos_count,
            "non_speeding_windows": total_count - pos_count,
            "speeding_ratio_pct": pos_ratio,
        }
        window_reports.append(stat)
        print(f"Saved {out_file.name}: X={X_cat.shape}, y={y_cat.shape}, speeding={pos_count}/{total_count} ({pos_ratio:.2f}%)")

    # Save summary report
    report_path = reports_dir / "windowing_summary.json"
    with open(report_path, "w") as f:
        json.dump(window_reports, f, indent=4)

    print("\n--- Windowing Summary ---")
    print(pd.DataFrame(window_reports).to_string(index=False))

if __name__ == "__main__":
    generate_windowed_datasets()
