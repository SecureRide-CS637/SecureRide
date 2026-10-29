#!/usr/bin/env python3
"""
SecureRide dual-mode detection (paper Sec. 5, Algorithm 1) and its evaluation (Sec. 6.2.4, 6.3, Figs. 8-9).

Supports:
  - backend="torch" (PyTorch 1D ResNet / GRU)
  - backend="keras" (TensorFlow ResNet / GRU)
  - backend="sklearn" (HistGradientBoosting baseline)

Usage:
  python dualmode.py --task sidewalk --backend torch
  python dualmode.py --task speeding --backend torch
  python dualmode.py --task sidewalk --backend sklearn

Evaluates 9 operating cases:
  Tau in {1.0, 1.5, 2.0} s
  Target Alpha in {0.85, 0.90, 0.95} (or custom)
Measures:
  - Detection accuracy (correct slots / 500)
  - Detection latency (buffering delay + execution time)
  - False positives (FP) and false negatives (FN)
  - Mode transition count
  - Individual buffering time and model execution time
"""
import argparse, os
import numpy as np, pandas as pd
import secureride_data as sd
from secureride_model import load_predictor

DT = 0.2                                   # s per slot (5 Hz BMS sampling)


# --------------------------------------------------------------------------- temporal smoothing (contribution 2)
def majority_vote(pred, n):
    """Majority vote of each verdict with the previous n-1 verdicts (causal, so it only ever adds delay).
    out[t] = 1  iff  more than half of the verdicts available in pred[t-n+1 .. t] are 1.
    n=1 returns pred unchanged. Use an ODD n (3, 5, 7, ...) to avoid ties. The first n-1 slots vote over
    the shorter history that exists."""
    pred = np.asarray(pred).astype(np.int64)
    if n <= 1:
        return pred.astype(np.int8)
    cs = np.r_[0, np.cumsum(pred)]
    idx = np.arange(len(pred))
    lo = np.maximum(idx - n + 1, 0)
    ones = cs[idx + 1] - cs[lo]
    cnt = idx - lo + 1
    return (2 * ones > cnt).astype(np.int8)


class MajorityVote:
    """Streaming version of majority_vote (what you would run on the scooter): push one verdict per slot."""
    def __init__(self, n):
        from collections import deque
        self.n, self.buf = n, deque(maxlen=max(n, 1))

    def push(self, verdict):
        self.buf.append(int(verdict))
        return int(2 * sum(self.buf) > len(self.buf))


# --------------------------------------------------------------------------- Algorithm 1
class DualModeDetector:
    """Streaming implementation of Algorithm 1 (paper Sec. 5.1).

    step(window_fn, n_avail) is called once per 0.2 s slot with a function:
      window_fn(L) -> (L,3) latest samples.
      Quick mode (default): run theta_quick on the last L_short samples.
        if illegal -> switch to Accurate mode; output = 1 immediately (low detection time).
      Accurate mode: run theta_accurate on the last L_long samples.
        The first accurate window must be built from the quick window + NEW samples (paper Sec. 5.2),
        so until L_long - L_short new slots have arrived, the provisional quick verdict (1) is held.
        if accurate says legal -> revert to Quick mode (quick detection was an error or illegal behavior ceased).
    """

    def __init__(self, quick, accurate, L_short, L_long, use_history=False):
        self.q, self.a, self.Ls, self.Ll, self.use_history = quick, accurate, L_short, L_long, use_history
        self.mode, self.t, self.t_enter = "quick", -1, None
        self.switch_count = 0

    def step(self, window_fn, n_avail):
        self.t += 1
        if self.mode == "quick":
            if n_avail < self.Ls:
                return 0
            y = int(self.q.predict(window_fn(self.Ls)[None])[0])
            if y == 1:
                self.mode = "accurate"
                self.t_enter = self.t
                self.switch_count += 1
            return y

        # Accurate mode
        new_since = self.t - self.t_enter + self.Ls
        if n_avail < self.Ll or (not self.use_history and new_since < self.Ll):
            return 1 # Hold provisional quick verdict while buffering
        y = int(self.a.predict(window_fn(self.Ll)[None])[0])
        if y == 0:
            self.mode = "quick"
            self.t_enter = None
            self.switch_count += 1
        return y


def slot_predictions(pred, X, L):
    """Prediction of `pred` on sliding window ending at every slot t."""
    out = np.zeros(len(X), np.int8)
    if len(X) >= L:
        W = np.lib.stride_tricks.sliding_window_view(X, L, axis=0).transpose(0, 2, 1) # (T-L+1, L, 3)
        out[L - 1:] = pred.predict(np.ascontiguousarray(W, dtype=np.float32))
    return out


def dual_from_slots(pq, pa, Ls, Ll, use_history=False):
    """Vectorized state machine matching DualModeDetector."""
    T = len(pq)
    out = np.zeros(T, np.int8)
    mode, t_enter = "quick", None
    switches = 0
    for t in range(T):
        if mode == "quick":
            y = pq[t] if t + 1 >= Ls else 0
            if y == 1:
                mode = "accurate"
                t_enter = t
                switches += 1
        else:
            new_since = t - t_enter + Ls
            if t + 1 < Ll or (not use_history and new_since < Ll):
                y = 1
            else:
                y = pa[t]
                if y == 0:
                    mode = "quick"
                    t_enter = None
                    switches += 1
        out[t] = y
    return out, switches


# --------------------------------------------------------------------------- model selection (Eq. 5, 6)
def select_models(acc, tau, alpha, hl_max_accurate=2, hl_quick=1, acc_col="val_acc", margin_ms=0.0):
    """acc: DataFrame with L, hl, val_acc, exec_ms.
    Quick    : HL = hl_quick, maximize L s.t.  exec_ms + L*0.2 s <= tau            (Eq. 5)
    Accurate : HL <= hl_max_accurate, minimize detection time s.t. accuracy >= alpha  (Eq. 6)
    Uses VALIDATION accuracy to ensure zero test split leakage.
    """
    a = acc.copy()
    a["t_det"] = a.L * DT + (a.exec_ms + margin_ms) / 1e3
    q = a[(a.hl == hl_quick) & (a.t_det <= tau)]
    if q.empty:
        q = a[a.hl == hl_quick].nsmallest(1, "L")
    q = q.sort_values("L").iloc[-1]

    c = a[(a.hl <= hl_max_accurate) & (a[acc_col] >= alpha)]
    if c.empty:
        r = a[a.hl <= hl_max_accurate].sort_values([acc_col, "L"], ascending=[False, True]).iloc[0]
    else:
        r = c.sort_values(["t_det", "hl"]).iloc[0]

    return (int(q.L), int(q.hl)), (int(r.L), int(r.hl))


# --------------------------------------------------------------------------- evaluation
def detection_time(out, ts, exec_ms, stable=3):
    """Seconds from illegal onset (ts) to first sustained detection run of `stable` slots.
    Includes buffering time + single-window inference execution time.
    """
    for t in range(ts, len(out) - stable + 1):
        if out[t:t + stable].all():
            buf_s = (t - ts + 1) * DT
            total_s = buf_s + exec_ms / 1e3
            return total_s, buf_s
    return float("nan"), float("nan")


EXT = {"torch": ".pt", "sklearn": ".joblib", "keras": ".keras"}


def evaluate_case(task, qkey, akey, scen, models_dir, acc, cache, arch="cnn", use_history=False, ts=100,
                  backend=None, smooth_q=1, smooth_a=1):
    """Evaluates Quick, Accurate, and Dual-Mode on all scenario trips.
    backend: load <stem>.<ext of that backend> (without it, load_predictor takes the first file it finds, which is
             .pt even when you asked for sklearn -- a bug if both kinds of file sit in models/).
    smooth_q / smooth_a: majority-vote window (slots) applied to the Quick / Accurate verdict streams BEFORE
             Algorithm 1 uses them (1 = off, the paper's behaviour)."""
    def get(key):
        model_name = f"{task}_L{key[0]}_HL{key[1]}"
        if arch != "cnn":
            model_name += f"_{arch}"
        if key not in cache["m"]:
            stem = f"{models_dir}/{model_name}"
            cache["m"][key] = load_predictor(stem + EXT[backend] if backend else stem)
        return cache["m"][key]

    def slots(key, i, X):
        k = (key, i)
        if k not in cache["s"]:
            cache["s"][k] = slot_predictions(get(key), X, key[0])
        return cache["s"][k]

    ex = lambda key: float(acc[(acc.L == key[0]) & (acc.hl == key[1])].exec_ms.iloc[0])

    res = {m: [] for m in ("quick", "accurate", "dual")}
    dt = {m: [] for m in res}
    buf_times = {m: [] for m in res}
    fp_counts = {m: [] for m in res}
    fn_counts = {m: [] for m in res}
    switches_list = []
    outs = []

    for i, (X, y) in enumerate(scen):
        pq = majority_vote(slots(qkey, i, X), smooth_q)
        pa = majority_vote(slots(akey, i, X), smooth_a)
        pd_, n_sw = dual_from_slots(pq, pa, qkey[0], akey[0], use_history)
        switches_list.append(n_sw)

        for m, o, e in (("quick", pq, ex(qkey)), ("accurate", pa, ex(akey)), ("dual", pd_, ex(qkey))):
            correct = int((o == y).sum())
            fp = int(((o == 1) & (y == 0)).sum())
            fn = int(((o == 0) & (y == 1)).sum())
            d_time, b_time = detection_time(o, ts, e)

            res[m].append(correct)
            fp_counts[m].append(fp)
            fn_counts[m].append(fn)
            dt[m].append(d_time)
            buf_times[m].append(b_time)

        outs.append((pq, pa, pd_))

    r = {f"{m}_correct": float(np.mean(v)) for m, v in res.items()}
    r.update({f"{m}_dt_s": float(np.nanmean(v)) for m, v in dt.items()})
    r.update({f"{m}_buf_s": float(np.nanmean(v)) for m, v in buf_times.items()})
    r.update({f"{m}_fp": float(np.mean(v)) for m, v in fp_counts.items()})
    r.update({f"{m}_fn": float(np.mean(v)) for m, v in fn_counts.items()})
    r["quick_exec_ms"] = ex(qkey)
    r["accurate_exec_ms"] = ex(akey)
    r["mean_switches"] = float(np.mean(switches_list))

    return r, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--task", choices=["speeding", "sidewalk"], required=True)
    ap.add_argument("--backend", choices=["torch", "keras", "sklearn"], default="torch")
    ap.add_argument("--arch", choices=["cnn", "gru"], default="cnn")
    ap.add_argument("--models", default="models")
    ap.add_argument("--results", default="results")
    ap.add_argument("--taus", nargs="+", type=float, default=[1.0, 1.5, 2.0], help="quick-mode time constraints (s)")
    ap.add_argument("--alphas", nargs="+", type=float, default=None,
                    help="accuracy targets for the accurate model")
    ap.add_argument("--hl-max-accurate", type=int, default=2)
    ap.add_argument("--use-history", action="store_true")
    ap.add_argument("--smooth-n", type=int, default=1,
                    help="majority vote over the last N Quick verdicts (odd N; 1 = off)")
    ap.add_argument("--smooth-n-acc", type=int, default=1,
                    help="majority vote over the last N Accurate verdicts (odd N; 1 = off)")
    a = ap.parse_args()

    tag = a.backend if a.arch == "cnn" else f"{a.backend}_{a.arch}"
    acc_path = f"{a.results}/accuracy_{tag}.csv"
    if not os.path.exists(acc_path):
        acc_path = f"{a.results}/metrics_{tag}.csv"

    if not os.path.exists(acc_path):
        raise FileNotFoundError(f"Accuracy table not found: {acc_path}. Run train.py first.")

    acc = pd.read_csv(acc_path)
    acc = acc[acc.task == a.task]

    if a.alphas is None:
        lo, hi = acc[acc.hl <= a.hl_max_accurate].val_acc.min(), acc[acc.hl <= a.hl_max_accurate].val_acc.max()
        # Spread targets across achievable validation accuracy
        a.alphas = [round(lo + f * (hi - lo), 4) for f in (0.35, 0.70, 0.95)]

    scen = sd.load_scenarios(a.task, root=a.root)
    cache = {"m": {}, "s": {}}
    rows, store = [], {}
    case_num = 0

    print(f"=== Running Dual-Mode Evaluation for '{a.task}' [{tag.upper()}] ===")
    print(f"Time constraints (Tau): {a.taus} s | Target Accuracies (Alpha): {a.alphas}")

    for tau in a.taus:
        for alpha in a.alphas:
            case_num += 1
            qk, ak = select_models(acc, tau, alpha, a.hl_max_accurate)
            r, outs = evaluate_case(a.task, qk, ak, scen, a.models, acc, cache, arch=a.arch, use_history=a.use_history,
                                    backend=a.backend, smooth_q=a.smooth_n, smooth_a=a.smooth_n_acc)
            r.update(case=case_num, tau=tau, alpha=alpha,
                     quick=f"SL{qk[0]}/HL{qk[1]}", accurate=f"SL{ak[0]}/HL{ak[1]}")
            rows.append(r)
            store[case_num] = outs
            print(f"Case {case_num}: Tau={tau:.1f}s, Alpha={alpha:.3f} | Quick: {r['quick']}, Accurate: {r['accurate']} | "
                  f"Correct/500: Q={r['quick_correct']:.1f}, A={r['accurate_correct']:.1f}, D={r['dual_correct']:.1f} | "
                  f"DetTime: Q={r['quick_dt_s']:.2f}s, A={r['accurate_dt_s']:.2f}s, D={r['dual_dt_s']:.2f}s | "
                  f"FP/FN: D_fp={r['dual_fp']:.1f}, D_fn={r['dual_fn']:.1f}")

    df = pd.DataFrame(rows)
    os.makedirs(a.results, exist_ok=True)
    sm = "" if (a.smooth_n <= 1 and a.smooth_n_acc <= 1) else f"_smooth{a.smooth_n}-{a.smooth_n_acc}"
    out_csv = f"{a.results}/dualmode_{a.task}_{tag}{sm}.csv"
    df.to_csv(out_csv, index=False)
    plot_fig9(df, store, scen, a.task, f"{a.results}/fig9_dualmode_{a.task}_{tag}{sm}.png")
    print(f"\nSaved dual-mode results to {out_csv}")


def plot_fig9(df, store, scen, task, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gain = df.dual_correct - df[["quick_correct", "accurate_correct"]].max(axis=1)
    best = int(df.case[gain.idxmax()])
    worst = int(df.case[gain.idxmin()])

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.85, 0.85])

    # Subplot 1: Correct slots
    ax1 = fig.add_subplot(gs[0, 0])
    w = 0.27
    x = df.case.to_numpy()
    ax1.bar(x - w, df["quick_correct"], w, color="#f2e94e", edgecolor="k", lw=.6, label="Quick")
    ax1.bar(x, df["accurate_correct"], w, color="#7ab51d", edgecolor="k", lw=.6, label="Accurate")
    ax1.bar(x + w, df["dual_correct"], w, color="#0b7a5a", edgecolor="k", lw=.6, label="Dual-Mode")
    lo = df[["quick_correct", "accurate_correct", "dual_correct"]].min().min()
    ax1.set_ylim(max(0, lo - 20), 505)
    ax1.set_xticks(x)
    ax1.set_xlabel("Operating Case #")
    ax1.set_ylabel("Correct Slots / 500")
    ax1.set_title(f"{task.capitalize()}: Correct Slots Comparison")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(ncol=3, fontsize=8)

    # Subplot 2: Detection time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - w, df["quick_dt_s"], w, color="#f2e94e", edgecolor="k", lw=.6, label="Quick")
    ax2.bar(x, df["accurate_dt_s"], w, color="#7ab51d", edgecolor="k", lw=.6, label="Accurate")
    ax2.bar(x + w, df["dual_dt_s"], w, color="#0b7a5a", edgecolor="k", lw=.6, label="Dual-Mode")
    ax2.set_xticks(x)
    ax2.set_xlabel("Operating Case #")
    ax2.set_ylabel("Detection Time (s)")
    ax2.set_title("Detection Time Comparison")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(ncol=3, fontsize=8)

    # Subplot 3 & 4: Representative timeline traces
    for row, (cs, nm) in enumerate(((best, "Best Advantage"), (worst, "Lowest Advantage")), start=1):
        axc = fig.add_subplot(gs[row, :])
        pq, pa, pd_ = store[cs][0]
        t = np.arange(len(pq))
        axc.axvspan(100, 400, color="#d0d0d0", alpha=0.4, label="Ground Truth Illegal (20s - 80s)")
        axc.plot(t, pq * 1.0 + 0.05, "-", color="#d4ac0d", lw=1.2, alpha=0.8, label="Quick Verdict")
        axc.plot(t, pa * 0.95, "--", color="#27ae60", lw=1.4, alpha=0.8, label="Accurate Verdict")
        axc.plot(t, pd_ * 0.85, "-", color="#196f3d", lw=1.8, label="Dual-Mode Output")
        r = df[df.case == cs].iloc[0]
        axc.set_title(f"Case #{cs} ({nm}): Quick={r.quick}, Accurate={r.accurate}, Tau={r.tau}s, Alpha={r.alpha:.3f}", fontsize=9)
        axc.set_yticks([0, 1])
        axc.set_yticklabels(["Legal", "Illegal"])
        axc.set_xlabel("Time (0.2 s slots)")
        axc.grid(True, linestyle=":", alpha=0.5)
        if row == 1:
            axc.legend(ncol=4, fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved dual-mode comparison plot to {out}")


if __name__ == "__main__":
    main()
