"""
Configuration for SecureRide Dual-Mode Detection (Online Stage).
Implements paper Section 5 and Algorithm 1 parameters.
"""

from pathlib import Path

DUAL_CONFIG = {
    "quick_model": "models/theta_quick.h5",       # SL5  HL1
    "accurate_model": "models/theta_accurate.h5", # SL20 HL3 (swapped from SL25 HL2)
    "scaler": "data/final/scaler.npz",
    "test_ride": "data/processed/ride_29_12_2021_13_21_21.npz",  # held-out ride
    "L_short": 5,
    "L_long": 20,
    "speed_limit_kmh": 20.0,
    "sampling_hz": 1.0,
    "results_dir": "results/dual_mode",
}
