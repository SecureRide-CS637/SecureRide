#!/usr/bin/env python3
"""
Calibration plots: does the simulated data show the two battery effects the paper builds on? (paper Fig. 2)

    python3 calibration_plots.py --root data --out results

  fig2a_speed_voltage.png       voltage distribution for speed < 10, 10-20, > 20 km/h        (paper Fig. 2a)
  fig2b_sidewalk_dv.png         dV = V(t+1)-V(t) for Road vs sidewalk Types 1-3               (paper Fig. 2b)
  fig2c_dv_all_types.png        extra: std of dV for road and all 13 sidewalk types
  calibration_summary.txt       the numbers to quote (honest comparison with the paper's 9.5 %)
Uses ONLY the training trips (no leakage of val/test into any claim) unless --all-splits is given.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data"); ap.add_argument("--out", default="results")
    ap.add_argument("--all-splits", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(f"{a.root}/trips.csv")
    df = df[df.split.isin(["train", "val", "test"] if a.all_splits else ["train"])].copy()
    df["speed"] = df.speed.clip(lower=0)                                  # a few simulated samples are slightly < 0
    df["dV"] = df.groupby("trip_id").voltage.diff().shift(-1)             # V(t+1) - V(t), never across trips
    df.loc[df.groupby("trip_id").cumcount(ascending=False) == 0, "dV"] = np.nan
    lines = [f"rows used: {len(df):,} ({'all splits' if a.all_splits else 'train trips only'})"]

    # ---------------- Fig 2a
    bins = [(0, 10, "Under 10"), (10, 20, "10-20"), (20, 99, "Over 20")]
    groups = [df.voltage[(df.speed >= lo) & (df.speed < hi)].to_numpy() for lo, hi, _ in bins]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    bp = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=.55)
    ax.set_xticklabels([b[2] for b in bins])
    for p in bp["boxes"]: p.set_facecolor("#6aa6a6")
    for m in bp["medians"]: m.set_color("k")
    ax.plot(range(1, 4), [g.mean() for g in groups], "o", color="red", ms=5, label="Mean")
    ax.set_xlabel("Speed (km/h)"); ax.set_ylabel("Voltage (V)"); ax.set_title("(a) Speeding violation"); ax.legend()
    ax.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{a.out}/fig2a_speed_voltage.png", dpi=170); plt.close()

    legal, fast = df.voltage[df.speed < 20], df.voltage[df.speed >= 20]
    lines += ["", "== Fig 2a: speed vs voltage =="]
    for (lo, hi, nm), g in zip(bins, groups):
        lines.append(f"  {nm:9s} n={len(g):7d}  mean {g.mean():.2f} V  median {np.median(g):.2f} V")
    drop = 100 * (1 - fast.mean() / legal.mean())
    lines.append(f"  voltage drop at >=20 km/h vs <20 km/h : {drop:.2f} %   (paper reports ~9.5 %)")
    # controlled version: the same charge level (capacity quintile), so SoC is not confounding the comparison
    df["socq"] = pd.qcut(df.capacity, 5, labels=False)
    ctrl = df.groupby("socq").apply(lambda g: 100 * (1 - g.voltage[g.speed >= 20].mean() / g.voltage[g.speed < 20].mean()),
                                    include_groups=False)
    lines.append(f"  same-charge-level drop (per capacity quintile): {', '.join(f'{v:.2f}' for v in ctrl)} %  (mean {ctrl.mean():.2f} %)")

    # ---------------- Fig 2b
    d = df.dropna(subset=["dV"]); rng = np.random.default_rng(0)
    cats = [("Road", 0), ("Type 1", 1), ("Type 2", 2), ("Type 3", 3)]
    fig, ax = plt.subplots(figsize=(5.2, 4)); cols = ["#444", "#e03a3a", "#1f5fd6", "#2aa86b"]
    for i, ((nm, s), c) in enumerate(zip(cats, cols)):
        v = d.dV[d.surface == s].to_numpy(); v = rng.choice(v, min(len(v), 400), replace=False)
        ax.scatter(i + rng.uniform(-.28, .28, len(v)), v, s=6, color=c, alpha=.7)
    ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in cats]); ax.set_xlabel("Road type")
    ax.set_ylabel(r"$\Delta V$ (V)"); ax.set_title("(b) Sidewalk riding"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig2b_sidewalk_dv.png", dpi=170); plt.close()

    lines += ["", "== Fig 2b: dV spread by surface =="]
    sd_by = d.groupby("surface").dV.std()
    for nm, s in cats: lines.append(f"  {nm:7s} std(dV) = {sd_by[s]:.3f} V   var = {sd_by[s]**2:.3f}")
    lines.append(f"  sidewalk (all 13 types) std {d.dV[d.surface > 0].std():.3f} V  vs road {d.dV[d.surface == 0].std():.3f} V "
                 f" -> ratio {d.dV[d.surface > 0].std() / d.dV[d.surface == 0].std():.2f}x")
    mono = np.corrcoef(sd_by.index[1:], sd_by.values[1:])[0, 1]
    lines.append(f"  type index vs std(dV) correlation over types 1-13: {mono:+.2f} "
                 f"(paper: smaller/more segmented blocks -> larger variance; only claim it if this is clearly negative AND "
                 f"your generator maps small blocks to low type numbers)")

    # ---------------- extra: all types
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(["Road"] + [f"T{i}" for i in range(1, 14)], sd_by.values, color=["#444"] + ["#2aa86b"] * 13)
    ax.set_ylabel(r"std($\Delta V$) (V)"); ax.set_title("Voltage jitter by surface type"); ax.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(f"{a.out}/fig2c_dv_all_types.png", dpi=170); plt.close()

    open(f"{a.out}/calibration_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
