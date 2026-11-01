import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG

import numpy as np
import pandas as pd

def determine_group_split(unique_rides, ride_id_array, target_split):
    """
    Assign whole rides to Train, Val, and Test to prevent data leakage across windows.
    Optimizes for split proportions (e.g. 64% train, 16% val, 20% test).
    """
    total_samples = len(ride_id_array)
    ride_counts = {r: int(np.sum(ride_id_array == r)) for r in unique_rides}

    # If exactly our 4 rides, choose optimal assignment:
    # 13_09_2021_14_51_57 (35.7%) + 13_09_2021_18_09_58 (34.7%) -> Train (70.4%)
    # 29_12_2021_17_48_30 (9.8%) -> Val (~10-16%)
    # 29_12_2021_13_21_21 (19.9%) -> Test (20.0%)
    if len(unique_rides) == 4 and "29_12_2021_13_21_21" in unique_rides:
        return {
            "train": ["13_09_2021_14_51_57", "13_09_2021_18_09_58"],
            "val": ["29_12_2021_17_48_30"],
            "test": ["29_12_2021_13_21_21"],
        }

    # General greedy / combinatorial search for best partition
    best_assignment = None
    best_error = float("inf")
    # For small number of rides, test partitions
    import itertools
    for train_r in itertools.combinations(unique_rides, max(1, len(unique_rides) - 2)):
        rem = [r for r in unique_rides if r not in train_r]
        for val_r in itertools.combinations(rem, max(1, len(rem) // 2)):
            test_r = [r for r in rem if r not in val_r]
            if not test_r:
                continue
            tr_frac = sum(ride_counts[r] for r in train_r) / total_samples
            val_frac = sum(ride_counts[r] for r in val_r) / total_samples
            te_frac = sum(ride_counts[r] for r in test_r) / total_samples
            err = (
                abs(tr_frac - target_split["train"]) ** 2
                + abs(val_frac - target_split["val"]) ** 2
                + abs(te_frac - target_split["test"]) ** 2
            )
            if err < best_error:
                best_error = err
                best_assignment = {
                    "train": list(train_r),
                    "val": list(val_r),
                    "test": list(test_r),
                }

    return best_assignment

def split_windowed_datasets():
    windowed_dir = CONFIG["paths"]["windowed"]
    reports_dir = CONFIG["paths"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    target_split = CONFIG["split"]
    seq_lengths = CONFIG["seq_lengths"]

    # Use reference dataset seq5 to determine group assignment
    ref_file = windowed_dir / "seq5.npz"
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference windowed file not found: {ref_file}. Run 05_window.py first.")

    ref_data = np.load(ref_file, allow_pickle=True)
    unique_rides = np.unique(ref_data["ride_id"])
    ride_assignment = determine_group_split(unique_rides, ref_data["ride_id"], target_split)

    print("Group-based Ride Assignment (Zero data leakage):")
    print(f"  Train rides ({len(ride_assignment['train'])}): {ride_assignment['train']}")
    print(f"  Val rides   ({len(ride_assignment['val'])}): {ride_assignment['val']}")
    print(f"  Test rides  ({len(ride_assignment['test'])}): {ride_assignment['test']}")

    class_balance_report = {
        "target_split": target_split,
        "ride_assignment": ride_assignment,
        "datasets": {},
    }

    split_summary_table = []

    for L in seq_lengths:
        data_path = windowed_dir / f"seq{L}.npz"
        data = np.load(data_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        ride_id = data["ride_id"]

        train_mask = np.isin(ride_id, ride_assignment["train"])
        val_mask = np.isin(ride_id, ride_assignment["val"])
        test_mask = np.isin(ride_id, ride_assignment["test"])

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Verify no ride_id appears in more than one split
        train_rides_set = set(np.unique(ride_id[train_mask]))
        val_rides_set = set(np.unique(ride_id[val_mask]))
        test_rides_set = set(np.unique(ride_id[test_mask]))
        assert train_rides_set.isdisjoint(val_rides_set), "Train and Val share ride IDs!"
        assert train_rides_set.isdisjoint(test_rides_set), "Train and Test share ride IDs!"
        assert val_rides_set.isdisjoint(test_rides_set), "Val and Test share ride IDs!"

        total_N = len(X)
        tr_pos = int(y_train.sum())
        val_pos = int(y_val.sum())
        te_pos = int(y_test.sum())

        tr_pos_ratio = tr_pos / len(y_train) if len(y_train) > 0 else 0
        val_pos_ratio = val_pos / len(y_val) if len(y_val) > 0 else 0
        te_pos_ratio = te_pos / len(y_test) if len(y_test) > 0 else 0

        # Check §8.1 imbalance recommendation: if positive < 15%
        oversampling_needed = tr_pos_ratio < 0.15

        ds_stat = {
            "seq_len": L,
            "total_samples": total_N,
            "train": {
                "count": len(y_train),
                "frac": len(y_train) / total_N,
                "positives": tr_pos,
                "negatives": len(y_train) - tr_pos,
                "pos_ratio": tr_pos_ratio,
            },
            "val": {
                "count": len(y_val),
                "frac": len(y_val) / total_N,
                "positives": val_pos,
                "negatives": len(y_val) - val_pos,
                "pos_ratio": val_pos_ratio,
            },
            "test": {
                "count": len(y_test),
                "frac": len(y_test) / total_N,
                "positives": te_pos,
                "negatives": len(y_test) - te_pos,
                "pos_ratio": te_pos_ratio,
            },
            "resampling_recommendation": "oversample_minority" if oversampling_needed else "none",
        }
        class_balance_report["datasets"][f"seq{L}"] = ds_stat

        split_summary_table.append({
            "seq": f"seq{L}",
            "N_train": len(y_train),
            "train_pos%": f"{tr_pos_ratio*100:.1f}%",
            "N_val": len(y_val),
            "val_pos%": f"{val_pos_ratio*100:.1f}%",
            "N_test": len(y_test),
            "test_pos%": f"{te_pos_ratio*100:.1f}%",
        })

        # Save split arrays for Phase 7 & 8
        out_split = windowed_dir / f"split_seq{L}.npz"
        np.savez_compressed(
            out_split,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        print(f"Saved {out_split.name}: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Save class balance report
    cb_path = reports_dir / "class_balance.json"
    with open(cb_path, "w") as f:
        json.dump(class_balance_report, f, indent=4)
    print(f"\nSaved class balance report to: {cb_path}")

    print("\n--- Split Summary Table ---")
    print(pd.DataFrame(split_summary_table).to_string(index=False))

if __name__ == "__main__":
    split_windowed_datasets()
