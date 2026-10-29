#!/usr/bin/env python3
"""
SecureRide -- Contribution 2: TEMPORAL SMOOTHING   (suggested by the paper as future work)

Every slot (0.2 s) a model gives a verdict. A single wrong verdict is a false alarm. A causal majority vote over
the last N verdicts removes isolated flickers, at the cost of ~ (N-1)/2 extra slots of detection delay.

What this script measures, for N in --ns (default 1 3 5 7 9 11; N=1 is "no smoothing" = the paper)
  A. FALSE ALARMS on ALL held-out test trips (natural class balance, ~144 trips)
       * legal zone   = slots whose last-L window is entirely legal   (nothing illegal happened)
       * illegal zone = slots whose last-L window is entirely illegal (paper's Eq. 2 positive)
       * windows that straddle a behaviour change are excluded (their label is ambiguous)
       * false-alarm slots/h, false-alarm EVENTS/h (rising edges 0->1 inside the legal zone),
         recall in the illegal zone, precision and F1 over the two zones
  B. COST: detection time on the 100 s scenario trips (illegal from 20 s to 80 s) and, for Dual mode,
     correct slots / 500, FP and FN.  Quick-only, Accurate-only and Dual are all reported.

Usage
  python3 smoothing.py --task sidewalk --backend torch                    # your CNN (models/*.pt)
  python3 smoothing.py --task speeding --backend torch --quick-L 5 --quick-hl 1 --acc-L 20 --acc-hl 2
  python3 smoothing.py --task sidewalk --backend sklearn                  # quick test without deep learning
Outputs
  results/smoothing_<backend>_<task>.csv     one row per (model, N)
  results/fig_smoothing_<backend>_<task>.png
"""
import argparse, os
import numpy as np, pandas as pd
import secureride_data as sd
from secureride_model import load_predictor
from dualmode import (DT, EXT, slot_predictions, dual_from_slots, majority_vote, detection_time)


# ---------------------------------------------------------------- data: held-out test trips
def load_trips(root, task, split="test"):
    """List of (X normalised (T,3), y per-timestep (T,)) for every trip of one split."""
    z = sd._npz(root)
    mean = np.array(z["_stats"]["mean"], np.float32); std = np.array(z["_stats"]["std"], np.float32)
    sel = np.flatnonzero(z["split"] == split)
    trips = z["trip"][sel]
    out = []
    for t in np.unique(trips):
        idx = sel[trips == t]
        X = z["series"][idx].astype(np.float32) / sd.SCALE.astype(np.float32)
        out.append(((X - mean) / std, z[task][idx].astype(np.int8)))
    return out


def zones(y, L):
    """legal[t] / illegal[t]: is the window of the last L slots (ending at t) entirely legal / entirely illegal."""
    T = len(y); cs = np.r_[0, np.cumsum(y.astype(np.int64))]
    legal = np.zeros(T, bool); ill = np.zeros(T, bool)
    if T >= L:
        s = cs[L:] - cs[:-L]                      # number of illegal slots in each window, ending at L-1 .. T-1
        legal[L - 1:] = s == 0
        ill[L - 1:] = s == L
    return legal, ill


# ---------------------------------------------------------------- metrics
class Acc:
    """Accumulates zone statistics over many trips."""
    def __init__(self):
        self.fa_slots = self.fa_events = self.legal_n = self.tp = self.ill_n = 0

    def add(self, out, legal, ill):
        o = out.astype(bool)
        self.legal_n += int(legal.sum()); self.ill_n += int(ill.sum())
        self.fa_slots += int((o & legal).sum()); self.tp += int((o & ill).sum())
        rising = o & ~np.r_[False, o[:-1]]
        self.fa_events += int((rising & legal).sum())

    def result(self):
        fp, tp = self.fa_slots, self.tp
        prec = tp / max(tp + fp, 1); rec = tp / max(self.ill_n, 1)
        hours = self.legal_n * DT / 3600.0
        return dict(fa_slot_rate=fp / max(self.legal_n, 1), fa_events_per_h=self.fa_events / max(hours, 1e-9),
                    recall=rec, precision=prec, f1=2 * prec * rec / max(prec + rec, 1e-9),
                    legal_slots=self.legal_n, illegal_slots=self.ill_n)


def get_model(models_dir, task, L, hl, backend, arch):
    name = f"{task}_L{L}_HL{hl}" + ("" if arch == "cnn" else f"_{arch}")
    return load_predictor(f"{models_dir}/{name}{EXT[backend]}")


# ---------------------------------------------------------------- experiment
def run(a):
    mq = get_model(a.models, a.task, a.quick_L, a.quick_hl, a.backend, a.arch)
    ma = get_model(a.models, a.task, a.acc_L, a.acc_hl, a.backend, a.arch)
    Lq, La = a.quick_L, a.acc_L
    names = {"quick": f"Quick SL{Lq}/HL{a.quick_hl}", "accurate": f"Accurate SL{La}/HL{a.acc_hl}", "dual": "Dual mode"}

    test = load_trips(a.root, a.task, "test")
    scen = sd.load_scenarios(a.task, root=a.root)
    print(f"{len(test)} test trips, {sum(len(y) for _, y in test)} slots | {len(scen)} scenario trips")

    # raw (unsmoothed) per-slot verdicts, computed once
    raw_t = [(slot_predictions(mq, X, Lq), slot_predictions(ma, X, La)) for X, _ in test]
    raw_s = [(slot_predictions(mq, X, Lq), slot_predictions(ma, X, La)) for X, _ in scen]
    exq, exa = mq.exec_ms(), ma.exec_ms()

    rows = []
    for n in a.ns:
        accs = {k: Acc() for k in names}
        for (X, y), (pq, pa) in zip(test, raw_t):
            sq, sa = majority_vote(pq, n), majority_vote(pa, n)
            sd_, _ = dual_from_slots(sq, sa, Lq, La, a.use_history)
            legal, ill = zones(y, La)                      # one common zone definition (longest window) for all three
            for k, o in (("quick", sq), ("accurate", sa), ("dual", sd_)):
                accs[k].add(o, legal, ill)

        scen_res = {k: dict(correct=[], fp=[], fn=[], dt=[]) for k in names}
        for (X, y), (pq, pa) in zip(scen, raw_s):
            sq, sa = majority_vote(pq, n), majority_vote(pa, n)
            sd_, _ = dual_from_slots(sq, sa, Lq, La, a.use_history)
            for k, o, e in (("quick", sq, exq), ("accurate", sa, exa), ("dual", sd_, exq)):
                scen_res[k]["correct"].append(int((o == y).sum()))
                scen_res[k]["fp"].append(int(((o == 1) & (y == 0)).sum()))
                scen_res[k]["fn"].append(int(((o == 0) & (y == 1)).sum()))
                scen_res[k]["dt"].append(detection_time(o, 100, e)[0])

        for k in names:
            r = accs[k].result()
            sr = scen_res[k]
            r.update(model=names[k], kind=k, N=n, scen_correct=float(np.mean(sr["correct"])),
                     scen_fp=float(np.mean(sr["fp"])), scen_fn=float(np.mean(sr["fn"])),
                     scen_detect_s=float(np.nanmean(sr["dt"])),
                     scen_missed=int(np.isnan(sr["dt"]).sum()))
            rows.append(r)
        d = [r for r in rows if r["N"] == n]
        print("N=%2d | " % n + " | ".join(f"{r['kind']}: FA/h={r['fa_events_per_h']:.0f} R={r['recall']:.2f} "
                                          f"F1={r['f1']:.2f} det={r['scen_detect_s']:.2f}s" for r in d))
    return pd.DataFrame(rows)


def plot(df, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"quick": "#d4ac0d", "accurate": "#27ae60", "dual": "#0b5345"}
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.9))
    spec = (("fa_events_per_h", "False-alarm events per hour\n(legal riding; lower is better)"),
            ("recall", "Recall in illegal zone"),
            ("f1", "F1 (legal + illegal zones)"),
            ("scen_detect_s", "Detection time on scenarios (s)\n(lower is better)"))
    for ax, (key, name) in zip(axs, spec):
        for kind, g in df.groupby("kind"):
            g = g.sort_values("N")
            ax.plot(g.N, g[key], "-o", color=col[kind], lw=2, ms=4, label=g.model.iloc[0])
        ax.set_xlabel("majority-vote window N (slots of 0.2 s; 1 = off)")
        ax.set_title(name, fontsize=9.5); ax.grid(True, linestyle="--", alpha=.4)
        ax.set_xticks(sorted(df.N.unique()))
    axs[0].legend(fontsize=7.5)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(); plt.savefig(out, dpi=170); plt.close()
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--task", choices=["speeding", "sidewalk"], default="sidewalk")
    ap.add_argument("--backend", choices=["torch", "keras", "sklearn"], default="torch")
    ap.add_argument("--arch", choices=["cnn", "gru"], default="cnn")
    ap.add_argument("--models", default="models")
    ap.add_argument("--results", default="results")
    ap.add_argument("--quick-L", type=int, default=5);  ap.add_argument("--quick-hl", type=int, default=1)
    ap.add_argument("--acc-L", type=int, default=20);   ap.add_argument("--acc-hl", type=int, default=2)
    ap.add_argument("--ns", nargs="+", type=int, default=[1, 3, 5, 7, 9, 11])
    ap.add_argument("--use-history", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.results, exist_ok=True)
    df = run(a)
    tag = f"{a.backend}_{a.task}" + ("" if a.arch == "cnn" else f"_{a.arch}")
    df.to_csv(f"{a.results}/smoothing_{tag}.csv", index=False)
    plot(df, f"{a.results}/fig_smoothing_{tag}.png",
         f"Temporal smoothing: {a.task} ({a.backend.upper()}), Quick SL{a.quick_L}/HL{a.quick_hl}, "
         f"Accurate SL{a.acc_L}/HL{a.acc_hl}")
    keep = ["kind", "N", "fa_events_per_h", "fa_slot_rate", "recall", "precision", "f1",
            "scen_detect_s", "scen_correct", "scen_fp", "scen_fn"]
    print("\n" + df[keep].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
