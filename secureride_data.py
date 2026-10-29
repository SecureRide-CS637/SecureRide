#!/usr/bin/env python3
"""
SecureRide dataset builder + loader  (Kim et al., EMSOFT 2025 / ACM TECS 24(5s))

Run once:      python3 secureride_data.py build  [--root data]
Then in code:  from secureride_data import load, augment, load_scenarios
               X, y = load("sidewalk", L=15, split="train", augment=True)   # X: (N, L, 3) float32, channels = [V, I, C]

Paper-faithful pieces implemented here
  * input  = sequences of voltage, current, capacity            (Sec. 3, 4.2.1)
  * output = binary illegal / legal, one predictor PER behaviour (speeding, sidewalk)
  * window label is TRUE only if ALL L points are illegal        (Sec. 4.1.2, Eq. 2)
  * moving window, step 1, sequence lengths 5..30                (Sec. 4.1.2, Fig. 5)
  * group-based 64/16/20 split by trip (done at generation time) (Sec. 6.1)
  * augmentation = Gaussian U spike U step/slope, battery-consistent  (Sec. 4.1.3, Eq. 3)
  * 50/50 class balance (paper Table 3 test sets are ~50/50; Sec. 8.1 says data were balanced)
"""
import argparse, json, os
import numpy as np
import pandas as pd

LENGTHS = [5, 7, 10, 15, 20, 25, 30]
TASKS = ("speeding", "sidewalk")
SPLITS = ("train", "val", "test")
SCALE = np.array([100.0, 100.0, 1.0])        # int16 storage: V,I in 0.01 units, C in mAh


# ---------------------------------------------------------------- build
def build(root="data", seed=0):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(f"{root}/trips.csv")
    df = df.sort_values(["trip_id", "t"], kind="stable").reset_index(drop=True)
    raw = df[["voltage", "current", "capacity"]].to_numpy(np.float64)
    series = np.rint(raw * SCALE).astype(np.int16)
    trip = df["trip_id"].to_numpy()
    split = df["split"].to_numpy().astype(str)
    lab = {"speeding": df["speeding"].to_numpy(np.int8), "sidewalk": df["sidewalk"].to_numpy(np.int8)}
    surface = df["surface"].to_numpy(np.int8)

    # normalisation constants from TRAIN timesteps only
    tr = raw[split == "train"]
    stats = {"mean": tr.mean(0).tolist(), "std": tr.std(0).tolist()}
    json.dump(stats, open(f"{root}/norm_stats.json", "w"), indent=1)

    out = dict(series=series, trip=trip.astype(np.int32), split=split, surface=surface,
               speeding=lab["speeding"], sidewalk=lab["sidewalk"])

    n = len(df)
    # window start k is valid if rows k..k+L-1 all belong to the same trip
    last_of_trip = np.r_[trip[1:] != trip[:-1], True]
    trip_end = np.empty(n, dtype=np.int64)              # index of last row of this row's trip
    e = n - 1
    for i in range(n - 1, -1, -1):
        if last_of_trip[i]: e = i
        trip_end[i] = e

    rows = []
    for task in TASKS:
        cs = np.r_[0, np.cumsum(lab[task], dtype=np.int64)]
        for L in LENGTHS:
            starts = np.arange(0, n - L + 1)
            ok = trip_end[starts] >= starts + L - 1
            starts = starts[ok]
            y = ((cs[starts + L] - cs[starts]) == L).astype(np.int8)       # all-true rule
            sp = split[starts]
            for s in SPLITS:
                m = sp == s
                st, yy = starts[m], y[m]
                pos, neg = st[yy == 1], st[yy == 0]
                k = min(len(pos), len(neg))                                   # balance 50/50
                pos = rng.choice(pos, k, replace=False); neg = rng.choice(neg, k, replace=False)
                st_b = np.concatenate([pos, neg]); y_b = np.r_[np.ones(k, np.int8), np.zeros(k, np.int8)]
                p = rng.permutation(len(st_b))
                out[f"{task}_L{L}_{s}_start"] = st_b[p].astype(np.int32)
                out[f"{task}_L{L}_{s}_y"] = y_b[p]
                rows.append(dict(task=task, L=L, split=s, windows_before_balance=int(m.sum()),
                                 pos_before=int(len(np.flatnonzero(yy == 1))), balanced_windows=int(2 * k)))
    np.savez_compressed(f"{root}/secureride_windows.npz", **out)
    rep = pd.DataFrame(rows)
    rep.to_csv(f"{root}/window_counts.csv", index=False)
    print(rep.pivot_table(index=["task", "L"], columns="split", values="balanced_windows").to_string())
    print("\nnormalisation:", stats)


# ---------------------------------------------------------------- load
_cache = {}
def _npz(root):
    if root not in _cache:
        _cache[root] = dict(np.load(f"{root}/secureride_windows.npz", allow_pickle=False))
        _cache[root]["_stats"] = json.load(open(f"{root}/norm_stats.json"))
    return _cache[root]

def _gather(z, starts, L):
    idx = starts[:, None] + np.arange(L)[None, :]
    return (z["series"][idx].astype(np.float32) / SCALE.astype(np.float32))        # (N, L, 3) raw units

def load(task, L, split, root="data", augment=False, include_original=True, normalize=True, seed=0):
    """Return X (N,L,3) float32 [voltage V, current A, capacity mAh] and y (N,) int8 (1 = illegal)."""
    z = _npz(root)
    X = _gather(z, z[f"{task}_L{L}_{split}_start"], L)
    y = z[f"{task}_L{L}_{split}_y"]
    if augment:
        assert split == "train", "augment the training split only"
        X, y = augment_set(X, y, include_original=include_original, seed=seed)
    if normalize:
        X = (X - np.array(z["_stats"]["mean"], np.float32)) / np.array(z["_stats"]["std"], np.float32)
    return X, y

def load_scenarios(task, root="data", normalize=True):
    """100 s scenarios (500 slots of 0.2 s, illegal from 20 s to 80 s) for dual-mode evaluation.
    Returns list of (X (500,3), y_per_timestep (500,))."""
    z = _npz(root)
    sel = z["split"] == f"scenario_{task}"
    trips = np.unique(z["trip"][sel])
    mean = np.array(z["_stats"]["mean"], np.float32); std = np.array(z["_stats"]["std"], np.float32)
    out = []
    for t in trips:
        m = z["trip"] == t
        X = z["series"][m].astype(np.float32) / SCALE.astype(np.float32)
        out.append(((X - mean) / std if normalize else X, z[task][m]))
    return out


# ---------------------------------------------------------------- augmentation (paper Sec. 4.1.3)
# Noise is injected in PHYSICAL units and kept battery-consistent: a current disturbance dI always
# comes with dV = -R*dI (paper Eq. 4), R ~ 0.1-0.3 ohm.  Capacity is left untouched (it is a slow integral).
def f_gaussian(X, rng, sv=0.05, si=0.15, sc=3.0):
    return X + rng.normal(0, 1, X.shape).astype(np.float32) * np.array([sv, si, sc], np.float32)

def f_spike(X, rng, max_spikes=2):
    """short sharp changes: throttle taps, bumps, manholes."""
    X = X.copy(); N, L, _ = X.shape
    R = rng.uniform(0.1, 0.3, N)                       # one R per window -> dV = -R*dI always holds
    for _ in range(max_spikes):
        use = rng.random(N) < 0.6
        pos = rng.integers(0, L, N)
        dI = rng.choice([-1, 1], N) * rng.uniform(1.0, 5.0, N) * use
        X[np.arange(N), pos, 1] += dI
        X[np.arange(N), pos, 0] -= R * dI
    return X

def f_step_slope(X, rng):
    """step = persistent power change (e.g. entering an incline); slope = linear ramp (gradual incline)."""
    X = X.copy(); N, L, _ = X.shape
    amp = rng.choice([-1, 1], N) * rng.uniform(0.5, 3.0, N)
    R = rng.uniform(0.1, 0.3, N)
    is_step = rng.random(N) < 0.5
    pos = rng.integers(1, max(L, 2), N)
    t = np.arange(L)[None, :]
    step = (t >= pos[:, None]).astype(np.float32)
    slope = (t / max(L - 1, 1)).astype(np.float32) * np.ones((N, 1), np.float32)
    shape = np.where(is_step[:, None], step, slope)
    dI = amp[:, None] * shape
    X[:, :, 1] += dI
    X[:, :, 0] -= R[:, None] * dI
    return X

def augment_set(X, y, include_original=True, seed=0):
    """D_tr = F_GN(D) U F_SN(D) U F_SS(D)   (+ original windows, assumption)."""
    rng = np.random.default_rng(seed)
    parts = [X] if include_original else []
    parts += [f_gaussian(X, rng), f_spike(X, rng), f_step_slope(X, rng)]
    n = len(parts)
    Xa = np.concatenate(parts, 0); ya = np.tile(y, n)
    p = rng.permutation(len(ya))
    return Xa[p], ya[p]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.root, a.seed)
