import json
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd

def recompute_metrics():
    sweep_path = ROOT_DIR / "results" / "sweep.csv"
    cb_path = ROOT_DIR / "reports" / "class_balance.json"
    out_path = ROOT_DIR / "results" / "metrics_corrected.csv"

    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep file not found at {sweep_path}")
    if not cb_path.exists():
        raise FileNotFoundError(f"Class balance file not found at {cb_path}")

    sweep = pd.read_csv(sweep_path)
    with open(cb_path, "r") as f:
        cb = json.load(f)

    rows = []
    for _, r in sweep.iterrows():
        L = int(r["L_seq"])
        ds = cb["datasets"][f"seq{L}"]
        pos = ds["test"]["positives"]
        neg = ds["test"]["negatives"]

        tp = round(float(r["tp_rate"]) * pos)
        fp = round(float(r["fp_rate"]) * neg)
        fn = round(float(r["fn_rate"]) * pos)
        tn = round(float(r["tn_rate"]) * neg)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Balanced accuracy = 0.5 * (Specificity + Sensitivity) = 0.5 * (TNR + TPR)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        bal_acc = 0.5 * (tnr + tpr)

        rows.append({
            "model": r["model"],
            "L_seq": L,
            "hl": int(r["hl"]),
            "acc": float(r["test_acc"]),
            "precision": float(precision),
            "recall": float(recall),
            "balanced_acc": float(bal_acc),
            "val_acc": float(r["val_acc"]),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_path, index=False)
    print(f"Saved corrected metrics to {out_path}\n")

    cols_to_print = ["model", "L_seq", "hl", "acc", "precision", "recall", "balanced_acc"]
    print(metrics_df[cols_to_print].round(4).to_string(index=False))

    # Cross check for Task D.2: cnn L20 hl3
    check_row = metrics_df[(metrics_df["model"] == "cnn") & (metrics_df["L_seq"] == 20) & (metrics_df["hl"] == 3)]
    if not check_row.empty:
        c = check_row.iloc[0]
        print("\n--- Task D.2 Hand-Check (CNN L20 HL3) ---")
        print(f"TP: {c['TP']}, FP: {c['FP']}, FN: {c['FN']}, TN: {c['TN']}")
        print(f"Precision: {c['precision']*100:.2f}% (Expected ~98.95%)")
        print(f"Recall:    {c['recall']*100:.2f}% (Expected ~97.19%)")
        print(f"Bal Acc:   {c['balanced_acc']*100:.2f}% (Expected ~96.86%)")

if __name__ == "__main__":
    recompute_metrics()
