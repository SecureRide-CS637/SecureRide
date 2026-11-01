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

def compute_labels():
    processed_dir = CONFIG["paths"]["processed"]
    reports_dir = CONFIG["paths"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    ride_files = sorted(list(processed_dir.glob("ride_*.npz")))
    if not ride_files:
        raise FileNotFoundError(f"No processed ride files found in {processed_dir}. Run 02_clean.py first.")

    speed_limit = CONFIG["speed_limit_kmh"]
    print(f"Applying speeding threshold: > {speed_limit} km/h")

    label_summary = []
    for f in ride_files:
        data = np.load(f)
        speed = data["speed"]
        point_label = (speed > speed_limit).astype(np.int8)

        # Update the ride npz file with point_label
        arrays = dict(data)
        arrays["point_label"] = point_label
        np.savez_compressed(f, **arrays)

        speeding_pts = int(point_label.sum())
        total_pts = len(point_label)
        speeding_pct = (speeding_pts / total_pts) * 100

        # Also inspect window-level label positive ratios across seq_lengths
        window_stats = {}
        for L in CONFIG["seq_lengths"]:
            wl = window_label(point_label, L)
            wl_pct = float(wl.mean() * 100) if len(wl) > 0 else 0.0
            window_stats[f"L{L}_pct"] = wl_pct

        stat = {
            "ride": f.name,
            "total_points": total_pts,
            "speeding_points": speeding_pts,
            "speeding_pct": speeding_pct,
            **window_stats,
        }
        label_summary.append(stat)
        print(f"Updated {f.name}: {speeding_pts}/{total_pts} speeding ({speeding_pct:.2f}%)")

    # Save labeling summary
    with open(reports_dir / "label_summary.json", "w") as f_out:
        json.dump(label_summary, f_out, indent=4)

    print("\n--- Labeling Summary Table ---")
    print(pd.DataFrame(label_summary).to_string(index=False))

if __name__ == "__main__":
    compute_labels()
