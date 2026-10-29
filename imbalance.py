#!/usr/bin/env python3
"""
SecureRide -- Contribution 1: CLASS IMBALANCE   (a limitation named in the paper, Sec. 8.1)

The shipped windows in data/secureride_windows.npz are balanced 50/50. Real riding is mostly legal, so
this script rebuilds the windows from the raw trips at a chosen illegal prevalence (50 %, 5 %, 1 % ...),
trains with and without class weights, and reports precision, recall, F1 (plus FPR, balanced accuracy,
average precision and the accuracy of a "always legal" classifier, to show why accuracy alone misleads).

Sampling design (per split: train / val / test)
  * all windows are rebuilt with the paper's all-true rule (Eq. 2): label = 1 only if ALL L slots are illegal
  * keep (up to --max-neg) legal windows, then subsample illegal windows so that  pos / (pos + neg) = prevalence
  * train, val AND test use the same prevalence (val/test must look like deployment; only train is re-weighted)
  * augmentation (Gaussian + spike + step/slope) is applied to the TRAIN split only, as in the main pipeline
  * extra rows "+thr": same model, but the decision threshold is tuned for F1 on the VALIDATION set
  * repeated over --seeds different subsamples; table reports mean and std

Usage
  python3 imbalance.py --backend torch   --task sidewalk --L 10 --hl 2         # your real CNN
  python3 imbalance.py --backend sklearn --task sidewalk --L 10 --hl 1         # quick, no deep-learning install
  python3 imbalance.py --backend torch --task speeding --L 20 --hl 2 --prevalences 0.5 0.05 0.01
Outputs
  results/imbalance_<backend>_<task>_L<L>_HL<hl>.csv   (one row per prevalence x weighting x seed)
  results/imbalance_<backend>_<task>_L<L>_HL<hl>_summary.csv
  results/fig_imbalance_<backend>_<task>_L<L>_HL<hl>.png
"""
import argparse, json, os, time
import numpy as np, pandas as pd
import secureride_data as sd
from secureride_model import make_predictor


# ---------------------------------------------------------------- window rebuilding (all natural windows)
def all_windows(root, task, L, split):
    """Every valid window of one split with its all-true label. Returns (starts, y)."""
    z = sd._npz(root)
    trip, spl, lab = z["trip"], z["split"], z[task].astype(np.int64)
    n = len(trip)
    last = np.r_[trip[1:] != trip[:-1], True]
    trip_end = np.empty(n, np.int64); e = n - 1
    for i in range(n - 1, -1, -1):
        if last[i]: e = i
        trip_end[i] = e
    cs = np.r_[0, np.cumsum(lab)]
    starts = np.arange(0, n - L + 1)
    starts = starts[trip_end[starts] >= starts + L - 1]
    starts = starts[spl[starts] == split]
    y = ((cs[starts + L] - cs[starts]) == L).astype(np.int8)
    return starts, y


def sample_prevalence(starts, y, prevalence, max_neg, rng):
    """Keep legal windows (<= max_neg) and just enough illegal ones to hit `prevalence`."""
    pos, neg = starts[y == 1], starts[y == 0]
    if len(neg) > max_neg:
        neg = rng.choice(neg, max_neg, replace=False)
    n_pos = int(round(prevalence / (1.0 - prevalence) * len(neg))) if prevalence < 1 else len(pos)
    n_pos = max(1, min(n_pos, len(pos)))
    if n_pos < len(pos):
        pos = rng.choice(pos, n_pos, replace=False)
    elif prevalence < 0.5:                        # not enough illegal windows: shrink negatives instead
        n_neg = int(round((1 - prevalence) / prevalence * len(pos)))
        neg = rng.choice(neg, min(n_neg, len(neg)), replace=False)
    st = np.concatenate([pos, neg]); yy = np.r_[np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)]
    p = rng.permutation(len(st))
    return st[p], yy[p]


def make_set(root, task, L, split, prevalence, max_neg, seed, augment=False):
    z = sd._npz(root)
    starts, y = all_windows(root, task, L, split)
    st, yy = sample_prevalence(starts, y, prevalence, max_neg, np.random.default_rng(seed))
    X = sd._gather(z, st, L)                                  # raw units (N, L, 3)
    if augment:
        X, yy = sd.augment_set(X, yy, include_original=True, seed=seed)
    stats = z["_stats"]
    X = (X - np.array(stats["mean"], np.float32)) / np.array(stats["std"], np.float32)
    return X.astype(np.float32), yy.astype(np.int64)


# ---------------------------------------------------------------- metrics
def metrics(y, pred, proba1):
    from sklearn.metrics import (precision_score, recall_score, f1_score, balanced_accuracy_score,
                                 average_precision_score, confusion_matrix)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0), fpr=fp / max(fp + tn, 1),
        bal_acc=balanced_accuracy_score(y, pred), accuracy=(tp + tn) / len(y),
        always_legal_acc=1.0 - float(y.mean()), avg_precision=average_precision_score(y, proba1),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn), n_test=len(y), n_test_pos=int(y.sum()))


def weight_arg(name):
    return None if name.startswith("none") else "balanced"


def tune_threshold(yva, pva):
    """Threshold maximising F1 on the VALIDATION set (never on test)."""
    from sklearn.metrics import f1_score
    grid = np.linspace(0.02, 0.98, 49)
    f1s = [f1_score(yva, (pva >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


# ---------------------------------------------------------------- experiment
def run(a):
    rows = []
    for prev in a.prevalences:
        for seed in range(a.seeds):
            Xtr, ytr = make_set(a.root, a.task, a.L, "train", prev, a.max_neg, seed, augment=not a.no_augment)
            Xva, yva = make_set(a.root, a.task, a.L, "val", prev, a.max_neg, seed)
            Xte, yte = make_set(a.root, a.task, a.L, "test", prev, a.max_neg, seed)
            for wname in a.weights:
                t0 = time.time()
                p = make_predictor(a.backend, a.L, a.hl, arch=a.arch, epochs=a.epochs, batch=a.batch,
                                   l2=a.l2, seed=seed, verbose=0, class_weight=weight_arg(wname))
                p.fit(Xtr, ytr, Xva, yva)
                proba = p.predict_proba(Xte)[:, 1]
                variants = [(wname, 0.5)]
                if a.tune_threshold:
                    variants.append((wname + "+thr", tune_threshold(yva, p.predict_proba(Xva)[:, 1])))
                for vname, thr in variants:
                    m = metrics(yte, (proba >= thr).astype(np.int8), proba)
                    m.update(prevalence=prev, weights=vname, threshold=thr, seed=seed, train_n=len(ytr),
                             train_pos=int(ytr.sum()), train_s=time.time() - t0)
                    rows.append(m)
                    print(f"prev={prev:5.1%} seed={seed} {vname:13s} thr={thr:.2f} | P={m['precision']:.3f} "
                          f"R={m['recall']:.3f} F1={m['f1']:.3f} FPR={m['fpr']:.3f} AP={m['avg_precision']:.3f} "
                          f"(acc={m['accuracy']:.3f} vs always-legal {m['always_legal_acc']:.3f})")
    return pd.DataFrame(rows)


def plot(summary, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prevs = sorted(summary.prevalence.unique(), reverse=True)
    weights = list(dict.fromkeys(summary.weights))
    cols = {"none": "#c0392b", "balanced": "#196f3d", "none+thr": "#e59866", "balanced+thr": "#7dcea0"}
    names = {"none": "baseline (no weights)", "balanced": "class weights",
             "none+thr": "baseline + tuned threshold", "balanced+thr": "class weights + tuned threshold"}
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.0))
    for ax, (key, name) in zip(axs, (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                                     ("fpr", "False-positive rate (lower is better)"))):
        w = 0.8 / len(weights)
        for j, wn in enumerate(weights):
            vals = [summary[(summary.prevalence == pv) & (summary.weights == wn)][key + "_mean"].iloc[0] for pv in prevs]
            errs = [summary[(summary.prevalence == pv) & (summary.weights == wn)][key + "_std"].iloc[0] for pv in prevs]
            ax.bar(np.arange(len(prevs)) + j * w - 0.4 + w / 2, vals, w, yerr=errs, capsize=2,
                   color=cols.get(wn, "#555"), edgecolor="k", lw=.5,
                   label=names.get(wn, wn))
        ax.set_xticks(range(len(prevs))); ax.set_xticklabels([f"{pv:.0%}" for pv in prevs])
        ax.set_xlabel("illegal share of windows"); ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1.02); ax.grid(True, axis="y", linestyle="--", alpha=.4)
    axs[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(); plt.savefig(out, dpi=170); plt.close()
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--backend", choices=["torch", "keras", "sklearn"], default="torch")
    ap.add_argument("--arch", choices=["cnn", "gru"], default="cnn")
    ap.add_argument("--task", choices=["speeding", "sidewalk"], default="sidewalk")
    ap.add_argument("--L", type=int, default=10, help="sequence length (slots of 0.2 s)")
    ap.add_argument("--hl", type=int, default=2)
    ap.add_argument("--prevalences", nargs="+", type=float, default=[0.5, 0.05, 0.01])
    ap.add_argument("--weights", nargs="+", choices=["none", "balanced"], default=["none", "balanced"])
    ap.add_argument("--no-tune-threshold", dest="tune_threshold", action="store_false",
                    help="skip the extra rows that tune the decision threshold on the validation set")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-neg", type=int, default=20000, help="max legal windows per split (speed knob)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    os.makedirs(a.results, exist_ok=True)

    df = run(a)
    tag = f"{a.backend}_{a.task}_L{a.L}_HL{a.hl}" + ("" if a.arch == "cnn" else f"_{a.arch}")
    df.to_csv(f"{a.results}/imbalance_{tag}.csv", index=False)
    keys = ["precision", "recall", "f1", "fpr", "bal_acc", "accuracy", "avg_precision"]
    g = df.groupby(["prevalence", "weights"])
    summ = g[keys].agg(["mean", "std"]); summ.columns = [f"{k}_{s}" for k, s in summ.columns]
    summ = summ.fillna(0.0).reset_index()
    summ["always_legal_acc"] = summ.prevalence.map(lambda p: 1 - p)
    summ.to_csv(f"{a.results}/imbalance_{tag}_summary.csv", index=False)
    print("\n" + summ[["prevalence", "weights", "precision_mean", "recall_mean", "f1_mean", "fpr_mean",
                       "avg_precision_mean", "accuracy_mean", "always_legal_acc"]].round(3).to_string(index=False))
    plot(summ, f"{a.results}/fig_imbalance_{tag}.png",
         f"Class imbalance: {a.task}, SL={a.L}, HL={a.hl} ({a.backend.upper()}), {a.seeds} seeds")


if __name__ == "__main__":
    main()
