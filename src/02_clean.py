import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG

import numpy as np
import pandas as pd

def clean_series(series, min_val, max_val):
    """
    Replace out-of-bound values with NaN, interpolate linearly, and clamp boundaries.
    """
    s = series.copy()
    # Mark out of bounds
    invalid = (s < min_val) | (s > max_val)
    if invalid.any():
        s[invalid] = np.nan
        s = s.interpolate(method="linear").bfill().ffill()
    # Final clamp to enforce exact physical range
    return s.clip(lower=min_val, upper=max_val)

def clean_and_align_rides():
    reports_dir = CONFIG["paths"]["reports"]
    processed_dir = CONFIG["paths"]["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    schema_map_file = reports_dir / "schema_map.json"
    if not schema_map_file.exists():
        raise FileNotFoundError(f"Schema map not found at {schema_map_file}. Run 01_inspect.py first.")

    with open(schema_map_file, "r") as f:
        schema_map = json.load(f)

    target_hz = CONFIG["target_hz"]
    period_ms = int(1000 / target_hz)
    max_seq_len = max(CONFIG["seq_lengths"])

    cleaning_report = []

    for filename, cols in schema_map.items():
        csv_path = CONFIG["paths"]["raw"] / filename
        print(f"\nProcessing {filename}...")
        df = pd.read_csv(csv_path)

        time_col = cols["time"]
        speed_col = cols["speed"]
        volt_col = cols["voltage"]
        curr_col = cols["current"]
        soc_col = cols["soc"]
        ride_id_base = cols.get("ride_id") or Path(filename).stem

        # Task 2.1 — Timestamps: parse to epoch seconds, sort, drop duplicates
        t_sec = df[time_col].astype(float) / 1000.0  # epoch seconds
        clean_df = pd.DataFrame({
            "t": t_sec,
            "speed": df[speed_col].astype(float),
            "voltage": df[volt_col].astype(float),
            "current": df[curr_col].astype(float),
            "soc": df[soc_col].astype(float),
        })

        clean_df = clean_df.sort_values("t").drop_duplicates(subset=["t"]).reset_index(drop=True)

        # Detect gaps > 5 * median
        dt = clean_df["t"].diff()
        median_dt = dt.median() if len(dt) > 1 and not np.isnan(dt.median()) else 1.0
        gap_threshold = 5.0 * median_dt
        gap_indices = np.where(dt > gap_threshold)[0]

        # Split into continuous segments
        split_points = [0] + list(gap_indices) + [len(clean_df)]
        segments = []
        for i in range(len(split_points) - 1):
            start_i = split_points[i]
            end_i = split_points[i + 1]
            seg = clean_df.iloc[start_i:end_i].copy()
            if len(seg) > 0:
                segments.append(seg)

        print(f"File {filename} split into {len(segments)} segment(s) (gap threshold: {gap_threshold:.1f}s)")

        for seg_idx, seg in enumerate(segments):
            seg_id = f"{ride_id_base}_seg{seg_idx}" if len(segments) > 1 else ride_id_base

            # Task 2.3 — Unit normalization
            # Voltage: if > 100, mV -> V
            if (seg["voltage"] > 100.0).any():
                seg["voltage"] = seg["voltage"] / 1000.0
            # Current: if magnitude > 50, mA -> A
            if (seg["current"].abs() > 50.0).any():
                seg["current"] = seg["current"] / 1000.0
            # SoC: if all in [0, 1] and max <= 1.0, * 100
            if seg["soc"].max() <= 1.0 and seg["soc"].max() > 0:
                seg["soc"] = seg["soc"] * 100.0

            # Task 2.4 — Outliers clamp and interpolation
            seg["voltage"] = clean_series(seg["voltage"], 20.0, 60.0)
            seg["soc"] = clean_series(seg["soc"], 0.0, 100.0)
            seg["current"] = clean_series(seg["current"], -30.0, 30.0)
            seg["speed"] = clean_series(seg["speed"], 0.0, 60.0)

            # Task 2.2 — Resample to fixed rate (1 Hz)
            # Create a regular datetime index
            seg["dt_index"] = pd.to_datetime(seg["t"], unit="s")
            seg = seg.set_index("dt_index").sort_index()

            resampled = (
                seg[["t", "speed", "voltage", "current", "soc"]]
                .resample(f"{period_ms}ms")
                .mean()
                .interpolate(method="linear")
                .dropna()
            )

            # Ensure t column is strictly uniform epoch seconds
            t_start = resampled["t"].iloc[0]
            resampled["t"] = t_start + np.arange(len(resampled)) * (1.0 / target_hz)

            # Task 2.5 — Missing/edge handling
            if len(resampled) < max_seq_len:
                print(f"Segment {seg_id} length ({len(resampled)}) < max_seq_len ({max_seq_len}). Discarding.")
                continue

            # Task 2.6 — Persist as ride_<id>.npz
            out_file = processed_dir / f"ride_{seg_id}.npz"
            np.savez_compressed(
                out_file,
                t=resampled["t"].values.astype(np.float32),
                speed=resampled["speed"].values.astype(np.float32),
                voltage=resampled["voltage"].values.astype(np.float32),
                current=resampled["current"].values.astype(np.float32),
                soc=resampled["soc"].values.astype(np.float32),
            )

            stat = {
                "ride_id": seg_id,
                "file": out_file.name,
                "num_points": len(resampled),
                "duration_sec": len(resampled) / target_hz,
                "speed_mean": float(resampled["speed"].mean()),
                "speed_max": float(resampled["speed"].max()),
                "voltage_mean": float(resampled["voltage"].mean()),
                "current_mean": float(resampled["current"].mean()),
                "soc_start": float(resampled["soc"].iloc[0]),
                "soc_end": float(resampled["soc"].iloc[-1]),
            }
            cleaning_report.append(stat)
            print(f"Saved {out_file.name} ({len(resampled)} points, {stat['duration_sec']:.1f}s)")

    # Save cleaning summary
    report_file = reports_dir / "cleaning_summary.json"
    with open(report_file, "w") as f:
        json.dump(cleaning_report, f, indent=4)

    print("\n--- Cleaning Summary ---")
    print(pd.DataFrame(cleaning_report).to_string(index=False))

if __name__ == "__main__":
    clean_and_align_rides()
