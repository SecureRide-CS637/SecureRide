import os
import random
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FINAL_DIR = BASE_DIR / "data" / "final"
MODELS_DIR = BASE_DIR / "models"
SWEEP_DIR = MODELS_DIR / "sweep"
RESULTS_DIR = BASE_DIR / "results"
CONFUSION_DIR = RESULTS_DIR / "confusion"
REPORTS_DIR = BASE_DIR / "reports"

for d in [MODELS_DIR, SWEEP_DIR, RESULTS_DIR, CONFUSION_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_CONFIG = {
    "num_classes": 2,
    "channels": 3,
    "channel_order": ["voltage", "current", "soc"],  # must match data pipeline
    "seq_lengths": [5, 10, 15, 20, 25],  # SL sweep (Fig. 5)
    "hidden_layers": [1, 2, 3],  # HL = number of residual blocks
    "filters": 32,
    "kernel": 3,
    "dropout": 0.3,  # paper §6.1
    "l2": 0.01,  # paper §6.1
    "batch_size": 32,
    "epochs": 200,  # capped; early stopping will fire sooner
    "learning_rate": 1e-3,
    "early_stop_patience": 10,  # paper §6.1
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-5,
    "seed": 42,
    # heterogeneous selection (Case A adapted to 1 Hz sampling)
    "sampling_hz": 1.0,
    "tau_constraint_s": 6.0,  # 5 s buffer + ~10 ms exec + headroom at 1 Hz
    "alpha_target": 0.98,     # High-accuracy regime requiring long sequence windows
    "hl_limits": {"quick": 1, "accurate": 3},
    "paths": {
        "base": BASE_DIR,
        "data_final": DATA_FINAL_DIR,
        "models": MODELS_DIR,
        "sweep": SWEEP_DIR,
        "results": RESULTS_DIR,
        "confusion": CONFUSION_DIR,
        "reports": REPORTS_DIR,
    },
}

def set_seed(s=42):
    """Global seeding for CPU determinism (best-effort)"""
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    try:
        import numpy as np
        np.random.seed(s)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(s)
    except ImportError:
        pass
