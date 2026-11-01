from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
WINDOWED_DIR = DATA_DIR / "windowed"
FINAL_DIR = DATA_DIR / "final"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure all target directories exist
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, WINDOWED_DIR, FINAL_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "target_hz": 1.0,                # M365 BLE ≈ 1 Hz
    "speed_limit_kmh": 20.0,         # speeding threshold (paper uses 20/25 km/h)
    "channels": ["voltage", "current", "soc"],   # feature order is FIXED
    "seq_lengths": [5, 10, 15, 20, 25],  # short→quick model, long→accurate model
    "window_step": 1,                # paper Eq.2 moving window, step default 1
    "split": {"train": 0.64, "val": 0.16, "test": 0.20},  # paper 64/16/20
    "augment": {
        "gaussian_sigma_rel": 0.02,  # σ = 2% of per-channel std
        "spike_frac": 0.01,          # % of points to spike
        "spike_amp_rel": 2.0,        # spike amplitude = 2× channel std
        "step_slope_frac": 0.3,      # % of windows to add a ramp
    },
    "paths": {
        "base": BASE_DIR,
        "raw": RAW_DIR,
        "processed": PROCESSED_DIR,
        "windowed": WINDOWED_DIR,
        "final": FINAL_DIR,
        "reports": REPORTS_DIR,
    }
}
