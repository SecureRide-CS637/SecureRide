import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG

import numpy as np
import pandas as pd

def build_features():
    processed_dir = CONFIG["paths"]["processed"]
    ride_files = sorted(list(processed_dir.glob("ride_*.npz")))

    if not ride_files:
        raise FileNotFoundError(f"No processed ride files found in {processed_dir}. Run 02_clean.py first.")

    channels = CONFIG["channels"]
    print(f"Building features with fixed channels: {channels}")

    stats = []
    for f in ride_files:
        data = np.load(f)
        t = data["t"]
        speed = data["speed"]
        voltage = data["voltage"]
        current = data["current"]
        soc = data["soc"]

        # Channel order is FIXED: [voltage, current, soc]
        channel_data = []
        for ch in channels:
            if ch == "voltage":
                channel_data.append(voltage)
            elif ch == "current":
                channel_data.append(current)
            elif ch == "soc":
                channel_data.append(soc)
            else:
                raise ValueError(f"Unknown channel: {ch}")

        features = np.stack(channel_data, axis=1).astype(np.float32)  # (T, 3)

        # Derived diagnostic feature: ΔV = V[t+1] - V[t] (Task 3.2)
        # For the last element, replicate previous diff
        delta_v = np.diff(voltage, prepend=voltage[0]).astype(np.float32)

        # Update the ride npz file with features, delta_v, while keeping original channels
        np.savez_compressed(
            f,
            t=t,
            speed=speed,
            voltage=voltage,
            current=current,
            soc=soc,
            features=features,
            delta_v=delta_v,
        )

        stat = {
            "ride": f.name,
            "T": len(features),
            "features_shape": str(features.shape),
            "voltage_mean": float(features[:, 0].mean()),
            "voltage_std": float(features[:, 0].std()),
            "current_mean": float(features[:, 1].mean()),
            "current_std": float(features[:, 1].std()),
            "soc_mean": float(features[:, 2].mean()),
            "soc_std": float(features[:, 2].std()),
            "delta_v_mean": float(delta_v.mean()),
            "delta_v_std": float(delta_v.std()),
        }
        stats.append(stat)
        print(f"Updated {f.name}: features shape = {features.shape}")

    print("\n--- Features Summary ---")
    print(pd.DataFrame(stats).to_string(index=False))

if __name__ == "__main__":
    build_features()
