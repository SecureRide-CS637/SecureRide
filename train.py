#!/usr/bin/env python3
"""
SecureRide Model Training & Comprehensive Evaluation Pipeline
(Kim et al., ACM TECS 2025 / EMSOFT 2025)

Supports:
  - ResNet-based 1D CNN with Channel Attention (paper Sec. 6.1)
  - Stacked GRU Baseline (paper Sec. 6.1)
  - HistGradientBoosting Baseline (paper fallback)
Across:
  - Tasks: 'speeding' and 'sidewalk'
  - Sequence lengths (SL): 5, 7, 10, 15, 20, 25, 30
  - Hidden layers (HL): 1, 2, 3

Outputs:
  - Saved model checkpoints under `models/` (.pt, .joblib, or .keras)
  - Full metrics CSV under `results/metrics_<tag>.csv`
  - Per-model confusion matrices under `results/confusion_matrices/`
  - Figure 5 accuracy plots under `results/fig5_accuracy_<tag>.png`
  - Training convergence curves under `results/fig_training_curves_<tag>.png`
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)
import secureride_data as sd
from secureride_model import make_predictor


def get_data(task, L, split, root, augment=False, max_n=None, seed=0):
    """Raw windows -> optional subsample -> optional augmentation (train only) -> normalise."""
    X, y = sd.load(task, L, split, root=root, normalize=False)
    if max_n and len(y) > max_n:
        idx = np.random.default_rng(seed).choice(len(y), max_n, replace=False)
        X, y = X[idx], y[idx]
    if augment:
        # Gaussian U Spike U Step/Slope (+ originals), physical units
        X, y = sd.augment_set(X, y, include_original=True, seed=seed)
    st = json.load(open(f"{root}/norm_stats.json"))
    X = (X - np.array(st["mean"], np.float32)) / np.array(st["std"], np.float32)
    return X.astype(np.float32), y.astype(np.int64)


def evaluate_split(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, bal_acc=bal_acc, tp=tp, fp=fp, tn=tn, fn=fn, cm=cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--backend", choices=["torch", "keras", "sklearn"], default="torch")
    ap.add_argument("--arch", choices=["cnn", "gru"], default="cnn")
    ap.add_argument("--tasks", nargs="+", default=["speeding", "sidewalk"])
    ap.add_argument("--hls", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--lengths", nargs="+", type=int, default=[5, 7, 10, 15, 20, 25, 30])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-train", type=int, default=None, help="subsample train windows BEFORE augmentation")
    ap.add_argument("--max-eval", type=int, default=None, help="subsample val/test windows")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="models")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()

    os.makedirs(a.models, exist_ok=True)
    os.makedirs(a.results, exist_ok=True)
    cm_dir = os.path.join(a.results, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    tag = a.backend if a.arch == "cnn" else f"{a.backend}_{a.arch}"
    csv_path = f"{a.results}/metrics_{tag}.csv"
    acc_csv_path = f"{a.results}/accuracy_{tag}.csv" # for backward compatibility with dualmode

    rows = []
    histories = {}

    print(f"=== Starting SecureRide Training Sweep: Backend={a.backend}, Arch={a.arch} ===")
    print(f"Tasks: {a.tasks} | Lengths: {a.lengths} | Hidden Layers: {a.hls}")

    for task in a.tasks:
        for L in a.lengths:
            print(f"\n--- Loading data for task='{task}', L={L} ---")
            Xtr, ytr = get_data(task, L, "train", a.root, not a.no_augment, a.max_train, a.seed)
            Xva, yva = get_data(task, L, "val", a.root, False, a.max_eval, a.seed)
            Xte, yte = get_data(task, L, "test", a.root, False, a.max_eval, a.seed)

            print(f"Dataset sizes: Train={len(ytr)} (pos={int((ytr==1).sum())}, neg={int((ytr==0).sum())}) | "
                  f"Val={len(yva)} | Test={len(yte)}")

            for hl in a.hls:
                model_name = f"{task}_L{L}_HL{hl}"
                if a.arch != "cnn":
                    model_name += f"_{a.arch}"
                model_stem = f"{a.models}/{model_name}"

                t0 = time.time()
                p = make_predictor(a.backend, L, hl, arch=a.arch, epochs=a.epochs, batch=a.batch,
                                   l2=a.l2, seed=a.seed, verbose=1)

                fit_hist = p.fit(Xtr, ytr, Xva, yva)
                train_time = time.time() - t0

                # Predictions
                val_preds = p.predict(Xva)
                test_preds = p.predict(Xte)

                # Metrics
                val_m = evaluate_split(yva, val_preds)
                test_m = evaluate_split(yte, test_preds)

                # Latency & Model sizing
                exec_ms = p.exec_ms()
                params = p.n_params()
                p.save(model_stem)

                # Find file path and size
                file_ext = ".pt" if a.backend == "torch" else (".joblib" if a.backend == "sklearn" else ".keras")
                saved_file = model_stem + file_ext
                size_kb = os.path.getsize(saved_file) / 1024.0 if os.path.exists(saved_file) else 0.0

                # Save confusion matrix CSV
                cm_df = pd.DataFrame(test_m["cm"], index=["Actual_Legal", "Actual_Illegal"],
                                     columns=["Pred_Legal", "Pred_Illegal"])
                cm_df.to_csv(f"{cm_dir}/cm_{tag}_{model_name}.csv")

                row = dict(
                    task=task, arch=a.arch, backend=a.backend, L=L, hl=hl,
                    val_acc=val_m["acc"], val_prec=val_m["prec"], val_rec=val_m["rec"],
                    val_f1=val_m["f1"], val_bal_acc=val_m["bal_acc"],
                    test_acc=test_m["acc"], test_prec=test_m["prec"], test_rec=test_m["rec"],
                    test_f1=test_m["f1"], test_bal_acc=test_m["bal_acc"],
                    test_tp=test_m["tp"], test_fp=test_m["fp"], test_tn=test_m["tn"], test_fn=test_m["fn"],
                    exec_ms=exec_ms, params=params, size_kb=size_kb, train_time_s=train_time,
                    train_n=len(ytr), val_n=len(yva), test_n=len(yte)
                )
                rows.append(row)
                if fit_hist:
                    histories[model_name] = fit_hist

                # Incremental CSV saving
                df = pd.DataFrame(rows)
                df.to_csv(csv_path, index=False)
                # Also save standard accuracy csv for dual-mode compatibility
                compat_df = df[["task", "L", "hl", "val_acc", "test_acc", "exec_ms", "params", "train_n"]]
                compat_df.to_csv(acc_csv_path, index=False)

                print(f"[{task} SL{L} HL{hl}] TrainTime: {train_time:.1f}s | ValAcc: {val_m['acc']:.4f} | "
                      f"TestAcc: {test_m['acc']:.4f} | F1: {test_m['f1']:.4f} | Latency: {exec_ms:.2f}ms | Size: {size_kb:.1f}KB")

    df = pd.DataFrame(rows)
    plot_fig5(df, f"{a.results}/fig5_accuracy_{tag}.png", tag)
    if histories:
        plot_training_curves(histories, f"{a.results}/fig_training_curves_{tag}.png")
    print(f"\nTraining complete. Full metrics saved to {csv_path}")


def plot_fig5(df, out, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = [t for t in ("speeding", "sidewalk") if t in set(df.task)]
    fig, axs = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 4.0), squeeze=False)
    sty = {1: ("--", "#3366cc", "o"), 2: ("-.", "#109618", "s"), 3: ("-", "#dc3912", "^")}

    for ax, t in zip(axs[0], tasks):
        for hl, g in df[df.task == t].groupby("hl"):
            g = g.sort_values("L")
            fmt = sty.get(hl, ("-", "#333", "o"))
            ax.plot(g.L, 100 * g.test_acc, linestyle=fmt[0], color=fmt[1], marker=fmt[2],
                    lw=2, ms=5, label=f"HL {hl}")
        ax.set_xlabel("Sequence Length (SL, 0.2s slots)")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(f"{'Speeding Violation' if t == 'speeding' else 'Sidewalk Riding'} ({tag.upper()})")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")
        ax.set_ylim(min(70.0, df.test_acc.min() * 100 - 2), 101.0)

    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved accuracy plot to {out}")


def plot_training_curves(histories, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not histories:
        return
    # Select representative models: SL5/HL1, SL10/HL2, SL20/HL2 for sidewalk and speeding
    rep_keys = [k for k in histories.keys() if any(sub in k for sub in ["L5_HL1", "L10_HL2", "L20_HL2"])][:6]
    if not rep_keys:
        rep_keys = list(histories.keys())[:6]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(rep_keys)))

    for k, col in zip(rep_keys, colors):
        h = histories[k]
        if "loss" in h and len(h["loss"]) > 0:
            epochs = range(1, len(h["loss"]) + 1)
            ax1.plot(epochs, h["loss"], color=col, linestyle="-", label=f"{k} train")
            if "val_loss" in h:
                ax1.plot(epochs, h["val_loss"], color=col, linestyle="--")
        if "accuracy" in h and len(h["accuracy"]) > 0:
            epochs = range(1, len(h["accuracy"]) + 1)
            ax2.plot(epochs, [100 * a for a in h["accuracy"]], color=col, linestyle="-", label=f"{k}")
            if "val_accuracy" in h:
                ax2.plot(epochs, [100 * a for a in h["val_accuracy"]], color=col, linestyle="--")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved training curves to {out}")


if __name__ == "__main__":
    main()
