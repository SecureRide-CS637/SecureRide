import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from train_config import TRAIN_CONFIG, set_seed

import numpy as np
import tensorflow as tf

def load_split(L):
    """
    Load final train, val, and test splits for sequence length L.
    Returns: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    final_dir = TRAIN_CONFIG["paths"]["data_final"] / f"seq{L}"
    if not final_dir.exists():
        raise FileNotFoundError(f"Final directory not found: {final_dir}. Run data pipeline first.")

    train_data = np.load(final_dir / "train.npz")
    val_data = np.load(final_dir / "val.npz")
    test_data = np.load(final_dir / "test.npz")

    return (
        (train_data["X"], train_data["y"]),
        (val_data["X"], val_data["y"]),
        (test_data["X"], test_data["y"]),
    )

def to_ds(X, y, batch_size=32, train=False):
    """
    Create tf.data.Dataset from numpy arrays.
    Shuffles if train=True.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if train:
        ds = ds.shuffle(buffer_size=len(X), seed=TRAIN_CONFIG["seed"], reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def get_class_weights(y):
    """
    Compute class weights for unbalanced binary labels.
    y: one-hot array (N, 2), where column 0 is normal and 1 is speeding.
    """
    pos = float(y[:, 1].sum())
    neg = float(y[:, 0].sum())
    total = pos + neg
    if pos == 0 or neg == 0:
        return {0: 1.0, 1: 1.0}
    # Standard inverse frequency weighting
    return {0: total / (2.0 * neg), 1: total / (2.0 * pos)}
