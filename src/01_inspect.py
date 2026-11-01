import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def inspect_raw_data():
    raw_dir = CONFIG["paths"]["raw"]
    reports_dir = CONFIG["paths"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(raw_dir.glob("*.csv")))
    print(f"Found {len(csv_files)} CSV files in {raw_dir}")

    schema_map = {}
    inspection_summary = []

    # Prepare figure for raw overview plots (1 row per ride or subplots)
    n_files = len(csv_files)
    fig, axes = plt.subplots(n_files, 4, figsize=(20, 3.5 * n_files), sharex=False)
    if n_files == 1:
        axes = np.expand_dims(axes, 0)

    for idx, csv_path in enumerate(csv_files):
        print(f"\n{'='*60}\nInspecting: {csv_path.name}\n{'='*60}")
        df = pd.read_csv(csv_path)
        print(f"Shape: {df.shape}")
        print("\nData Types:")
        print(df.dtypes)
        print("\nHead (3 rows):")
        print(df.head(3))
        print("\nDescribe summary:")
        print(df.describe())

        # Column identification
        col_map = {}
        for c in df.columns:
            clow = c.lower().strip()
            if "millisecond" in clow or "epoch" in clow or clow == "time" or clow == "timestamp":
                col_map["time"] = c
            elif clow == "speed":
                col_map["speed"] = c
            elif "voltage" in clow:
                col_map["voltage"] = c
            elif "ampere" in clow or clow == "current":
                col_map["current"] = c
            elif "percentage" in clow or "soc" in clow or "battery_level" in clow:
                col_map["soc"] = c
        col_map["ride_id"] = csv_path.stem
        schema_map[csv_path.name] = col_map

        # Check required columns
        req = ["time", "speed", "voltage", "current", "soc"]
        missing_req = [r for r in req if r not in col_map]
        if missing_req:
            raise ValueError(f"Missing required columns {missing_req} in {csv_path.name}")

        time_col = col_map["time"]
        speed_col = col_map["speed"]
        volt_col = col_map["voltage"]
        curr_col = col_map["current"]
        soc_col = col_map["soc"]

        # Missing percentages
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        print(f"\nMissing values (%): {missing_pct}")

        # Timestamps
        t_sec = df[time_col] / 1000.0  # ms to s
        diff_t = t_sec.diff().dropna()
        time_span = t_sec.iloc[-1] - t_sec.iloc[0] if len(t_sec) > 1 else 0

        file_stat = {
            "file": csv_path.name,
            "rows": len(df),
            "time_span_seconds": float(time_span),
            "sampling_dt_median": float(diff_t.median()) if len(diff_t) > 0 else 0,
            "sampling_dt_min": float(diff_t.min()) if len(diff_t) > 0 else 0,
            "sampling_dt_max": float(diff_t.max()) if len(diff_t) > 0 else 0,
            "voltage_range": [float(df[volt_col].min()), float(df[volt_col].max())],
            "current_range": [float(df[curr_col].min()), float(df[curr_col].max())],
            "soc_range": [float(df[soc_col].min()), float(df[soc_col].max())],
            "speed_range": [float(df[speed_col].min()), float(df[speed_col].max())],
            "speeding_ratio_pct": float((df[speed_col] > CONFIG["speed_limit_kmh"]).mean() * 100),
        }
        inspection_summary.append(file_stat)

        # Plot signals
        rel_time = t_sec - t_sec.iloc[0]
        # Speed
        axes[idx, 0].plot(rel_time, df[speed_col], color="tab:blue", lw=1)
        axes[idx, 0].axhline(CONFIG["speed_limit_kmh"], color="red", linestyle="--", alpha=0.7, label=f"Limit ({CONFIG['speed_limit_kmh']} km/h)")
        axes[idx, 0].set_ylabel(f"{csv_path.stem[:10]}\nSpeed (km/h)")
        axes[idx, 0].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx, 0].set_title("Speed (km/h)")
        axes[idx, 0].legend(loc="upper right", fontsize="small")

        # Voltage
        axes[idx, 1].plot(rel_time, df[volt_col], color="tab:orange", lw=1)
        axes[idx, 1].set_ylabel("Voltage (V)")
        axes[idx, 1].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx, 1].set_title("Battery Voltage (V)")

        # Current
        axes[idx, 2].plot(rel_time, df[curr_col], color="tab:green", lw=1)
        axes[idx, 2].set_ylabel("Current (A)")
        axes[idx, 2].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx, 2].set_title("Battery Current (A)")

        # SoC
        axes[idx, 3].plot(rel_time, df[soc_col], color="tab:purple", lw=1)
        axes[idx, 3].set_ylabel("SoC (%)")
        axes[idx, 3].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx, 3].set_title("Battery SoC (%)")

        for col_i in range(4):
            axes[idx, col_i].set_xlabel("Time (s)")

    plt.tight_layout()
    overview_plot = reports_dir / "raw_overview.png"
    plt.savefig(overview_plot, dpi=150)
    plt.close()
    print(f"\nSaved raw overview plot to: {overview_plot}")

    # Save schema map JSON
    schema_map_path = reports_dir / "schema_map.json"
    with open(schema_map_path, "w") as f:
        json.dump(schema_map, f, indent=4)
    print(f"Saved schema map to: {schema_map_path}")

    # Save inspection report JSON
    insp_path = reports_dir / "inspection_summary.json"
    with open(insp_path, "w") as f:
        json.dump(inspection_summary, f, indent=4)
    print(f"Saved inspection summary to: {insp_path}")

    print("\n--- Inspection Summary Table ---")
    summary_df = pd.DataFrame(inspection_summary)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    inspect_raw_data()
