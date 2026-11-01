import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def gaussian(x, sigma_rel=0.02):
    """
    Paper F_GN: Gaussian noise, sigma = sigma_rel * per-channel std.
    Applied per-channel across the whole batch.
    """
    sigma = sigma_rel * x.std(axis=(0, 1), keepdims=True)  # (1, 1, 3)
    noise = np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return (x + noise).astype(np.float32)

def spike(x, frac=0.01, amp_rel=2.0):
    """
    Paper F_SN: Spike noise, sharp transient at random points.
    """
    x_out = x.copy()
    N, L, C = x_out.shape
    n_pts = max(1, int(frac * L))
    
    # Per-window per-channel std
    ch_std = x_out.std(axis=1)  # (N, C)
    # Avoid zero std edge case
    ch_std = np.where(ch_std == 0, 1e-4, ch_std)

    for i in range(N):
        idx = np.random.choice(L, n_pts, replace=False)
        # Random normal transient scaled by 2.0 * channel std
        noise = np.random.normal(0.0, amp_rel * ch_std[i], size=(n_pts, C))
        x_out[i, idx] += noise.astype(np.float32)
    return x_out.astype(np.float32)

def step_slope(x, frac=0.3):
    """
    Paper F_SS: Step & slope noise, gradual ramp added to a subset of windows.
    """
    x_out = x.copy()
    N, L, C = x_out.shape
    n_windows = int(frac * N)
    chosen_windows = np.random.choice(N, n_windows, replace=False)

    for i in chosen_windows:
        k = np.random.choice(C)  # select a channel
        c_std = x_out[i, :, k].std()
        if c_std == 0:
            c_std = 1e-4
        ramp = np.linspace(0.0, 0.05 * c_std, L, dtype=np.float32)
        direction = np.random.choice([-1.0, 1.0])
        x_out[i, :, k] += (ramp * direction).astype(np.float32)
    return x_out.astype(np.float32)

def augment_training_data():
    np.random.seed(42)
    windowed_dir = CONFIG["paths"]["windowed"]
    reports_dir = CONFIG["paths"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    seq_lengths = CONFIG["seq_lengths"]
    aug_cfg = CONFIG["augment"]

    aug_stats = []

    example_plotted = False

    for L in seq_lengths:
        split_file = windowed_dir / f"split_seq{L}.npz"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file {split_file} not found. Run 06_split.py first.")

        data = np.load(split_file)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        # Generate the 3 augmentations (TRAIN features only)
        X_gn = gaussian(X_train, sigma_rel=aug_cfg["gaussian_sigma_rel"])
        X_sn = spike(X_train, frac=aug_cfg["spike_frac"], amp_rel=aug_cfg["spike_amp_rel"])
        X_ss = step_slope(X_train, frac=aug_cfg["step_slope_frac"])

        # Combine: D_tr_final = D_train ∪ F_GN(D) ∪ F_SN(D) ∪ F_SS(D)
        X_train_aug = np.concatenate([X_train, X_gn, X_sn, X_ss], axis=0).astype(np.float32)
        y_train_aug = np.concatenate([y_train, y_train, y_train, y_train], axis=0).astype(np.int8)

        # Shuffle augmented train set
        perm = np.random.permutation(len(X_train_aug))
        X_train_aug = X_train_aug[perm]
        y_train_aug = y_train_aug[perm]

        # Task 7.5 — Sanity check plot on L=25 (or first sequence length)
        if not example_plotted and L >= 20:
            sample_idx = 10  # pick a sample with interesting signal dynamics
            orig_win = X_train[sample_idx]
            gn_win = X_gn[sample_idx]
            sn_win = X_sn[sample_idx]
            ss_win = X_ss[sample_idx]
            lbl = y_train[sample_idx]

            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            ch_names = ["Voltage (V)", "Current (A)", "SoC (%)"]
            colors = ["tab:orange", "tab:green", "tab:purple"]

            time_steps = np.arange(L)
            for c in range(3):
                axes[c].plot(time_steps, orig_win[:, c], "k-o", label="Original", lw=2, markersize=4)
                axes[c].plot(time_steps, gn_win[:, c], "--", color="tab:blue", label="Gaussian (F_GN)", lw=1.5)
                axes[c].plot(time_steps, sn_win[:, c], "-.", color="tab:red", label="Spike (F_SN)", lw=1.5)
                axes[c].plot(time_steps, ss_win[:, c], ":", color="tab:green", label="Step+Slope (F_SS)", lw=1.5)
                axes[c].set_ylabel(ch_names[c])
                axes[c].grid(True, alpha=0.3)
                if c == 0:
                    axes[c].set_title(f"Data Augmentation Comparison (L={L}, Label={'Speeding (1)' if lbl==1 else 'Normal (0)'})")
                    axes[c].legend(loc="upper right", fontsize="small")

            axes[2].set_xlabel("Time step within window (s)")
            plt.tight_layout()
            plot_path = reports_dir / "augment_example.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved augmentation example plot to: {plot_path}")
            example_plotted = True

        # Save augmented training set alongside unchanged val/test
        out_file = windowed_dir / f"augmented_seq{L}.npz"
        np.savez_compressed(
            out_file,
            X_train=X_train_aug,
            y_train=y_train_aug,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

        stat = {
            "seq_len": L,
            "orig_train_N": len(X_train),
            "aug_train_N": len(X_train_aug),
            "val_N": len(X_val),
            "test_N": len(X_test),
            "multiplier": len(X_train_aug) / len(X_train),
            "speeding_ratio_aug_train": float(y_train_aug.mean() * 100),
        }
        aug_stats.append(stat)
        print(f"Saved {out_file.name}: Train {len(X_train)} -> {len(X_train_aug)} (4x), Val={len(X_val)}, Test={len(X_test)}")

    # Save summary
    with open(reports_dir / "augmentation_summary.json", "w") as f:
        json.dump(aug_stats, f, indent=4)

    print("\n--- Augmentation Summary ---")
    print(pd.DataFrame(aug_stats).to_string(index=False))

if __name__ == "__main__":
    augment_training_data()
