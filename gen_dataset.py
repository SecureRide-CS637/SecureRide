#!/usr/bin/env python3
"""
Synthetic e-scooter battery dataset for re-implementing SecureRide (Kim et al., EMSOFT 2025 / ACM TECS 24(5s)).

Physics chain (per 0.2 s sample, matching the paper's 0.2 s slot / 5 Hz battery telemetry):
  rider throttle/brake -> speed profile -> wheel force (inertia + rolling + grade + drag + surface vibration)
  -> battery power (drive efficiency, regen, idle) -> current from  P = V*I
  -> terminal voltage  V = OCV(SoC) - I*R_internal(SoC,T) - V_rc      (paper Eq. 4, plus a 1st-order RC lag)
  -> remaining capacity by coulomb counting (the paper's 3rd battery factor).

Paper-derived design choices (all editable at the top):
  * Xiaomi M365-like pack: 10S Li-ion, ~42 V full, 7.8 Ah, max speed 25 km/h.
  * 36 h of driving, 13 sidewalk paving patterns + bike road, road share 53.09 % (paper Table 1).
  * Speed-range distribution 0-5/5-10/10-15/15-20/20-25 km/h = 3.10/15.98/23.66/20.60/36.66 % (Table 2).
  * Speeding label: measured speed > 20 km/h.  Sidewalk label: surface != road.
  * Varied SoC, slopes (flat/up/down), rider style, temperature, rider mass.
  * Group-based 64/16/20 split by trip (no window leakage).
  * Extra 100 s evaluation scenarios (illegal from 20 s to 80 s) for dual-mode evaluation (paper Sec. 6.3).
"""
import argparse, os
import numpy as np
import pandas as pd

DT = 0.2                     # s, 5 Hz  (paper: 0.2 s slots; SL5 -> 1.01 s detection)
G = 9.81
V_MAX = 25 / 3.6             # M365 limit
DIP_SCALE = 0.5              # multiplies rider brake-event rate
JIT_SCALE = 1.0              # multiplies rider speed jitter
GRADE_SCALE = 1.0            # multiplies all road slopes (0 = flat)
SPEED_LIMIT_KMH = 20.0       # speeding threshold used for labels (paper Fig. 2 / Sec. 2.3)

# ---------------- battery ----------------
SOC_PTS = np.linspace(0, 1, 11)
OCV_CELL = np.array([3.00, 3.45, 3.55, 3.62, 3.67, 3.73, 3.80, 3.88, 3.97, 4.07, 4.18])
def ocv(soc):                # 10S pack
    return 10.0 * np.interp(soc, SOC_PTS, OCV_CELL)

Q_NOM_AH = 7.8
R0_BASE = 0.16               # ohm, pack internal resistance at 25C, full SoC (varies per trip)
R1, TAU_RC = 0.05, 8.0       # 1st-order polarisation branch
ETA_DRIVE, ETA_REGEN = 0.80, 0.55
P_IDLE, P_REGEN_MAX, P_MAX = 4.0, 100.0, 450.0
CDA, RHO = 0.65, 1.2         # upright rider drag area, air density

# ---------------- surfaces (paper Table 1 / 2) ----------------
SURF_W = np.array([53.092, 3.662, 3.595, 5.645, 5.594, 5.558, 5.023, 4.800,
                   2.287, 2.187, 2.229, 2.156, 2.096, 2.077])
SURF_P = SURF_W / SURF_W.sum()          # index 0 = bike road, 1..13 = sidewalk types
SPEED_BINS = [(1.0, 5), (5, 10), (10, 15), (15, 19), (21.5, 24.5)]
SPEED_P = np.array([3.10, 15.98, 23.66, 20.60, 36.66]); SPEED_P /= SPEED_P.sum()

_prng = np.random.default_rng(12345)    # fixed: the 13 sidewalk types are the same in every run
BLOCK_LEN = _prng.uniform(0.06, 0.35, 13)             # m ; smaller blocks -> rougher (paper Sec 2.3)
VIB_SIGMA = np.concatenate([[1.5], 30.0 / np.sqrt(BLOCK_LEN / 0.1) * _prng.uniform(0.85, 1.15, 13)])  # N
CRR_EXTRA = np.concatenate([[0.0], 0.004 + 0.008 * _prng.random(13)])
SPIKE_P = np.concatenate([[0.002], np.full(13, 0.02)])  # per-sample prob. of a bump/joint/manhole


def pick_speed(rng):
    b = SPEED_BINS[rng.choice(5, p=SPEED_P)]
    return rng.uniform(*b)

def pick_grade(rng):
    """City bike lanes / sidewalks: 70 % flat, the rest gentle slopes of 0.5-1.5 % (up or down)."""
    if rng.random() < 0.7: return rng.uniform(-0.003, 0.003)
    return rng.choice([-1, 1]) * rng.uniform(0.005, 0.015)


def simulate(segments, rng, trip_id, calm=False):
    """segments: list of (duration_s, surface_idx, target_speed_kmh, grade). Returns (DataFrame, meta dict)."""
    mass = rng.uniform(55, 95) + 12.5
    temp = rng.uniform(5, 35)
    soh = rng.uniform(0.92, 1.0)
    r0 = R0_BASE * rng.uniform(0.8, 1.4) * np.exp(0.012 * (25 - temp))
    soc = rng.uniform(0.25, 1.0)
    q_as = Q_NOM_AH * soh * 3600.0
    q_est_as = soc * q_as * (1 + rng.normal(0, 0.015))       # BMS initial estimate error
    i_off = rng.normal(0, 0.03)
    a_acc, a_brk = rng.uniform(0.7, 1.3), rng.uniform(1.0, 2.0)
    jitter = rng.uniform(0.3, 1.2) * JIT_SCALE               # km/h OU std (rider smoothness)
    p_dip = rng.uniform(0.01, 0.04) * DIP_SCALE              # brake-event rate per sample
    crr0 = rng.uniform(0.010, 0.015)
    if calm:                                                 # steady rider (evaluation scenarios)
        jitter, p_dip = min(jitter, 0.4), 0.002

    v = segments[0][2] / 3.6 * rng.uniform(0.5, 1.0)
    ou = 0.0; v1 = 0.0; ar = 0.0; dip = 0; dip_amt = 0.0
    rows = []
    t = 0
    for dur, s, vt_kmh, grade in segments:
        for _ in range(int(round(dur / DT))):
            ou += -0.1 * ou + jitter / 3.6 * 0.3 * rng.normal()
            vt = vt_kmh / 3.6 + ou
            if dip > 0:
                vt -= dip_amt; dip -= 1
            elif rng.random() < p_dip:
                dip = rng.integers(5, 15); dip_amt = rng.uniform(3, 8) / 3.6
            vt = np.clip(vt, 0, V_MAX)
            a = np.clip(1.0 * (vt - v), -a_brk, a_acc)
            v_new = np.clip(v + a * DT, 0, V_MAX)
            a = (v_new - v) / DT; v = v_new

            # wheel force
            crr = crr0 + CRR_EXTRA[s]
            F = mass * a + mass * G * (crr + grade) + 0.5 * RHO * CDA * v * v
            # surface vibration (colored noise + occasional spikes)
            ar = 0.4 * ar + np.sqrt(1 - 0.16) * rng.normal()
            fv = 0.3 + min(v, 7.0) / 5.0
            F += VIB_SIGMA[s] * fv * ar
            if rng.random() < SPIKE_P[s]:
                F += rng.choice([-1, 1]) * rng.uniform(15, 45) * (1 if s else 0.5)

            Pm = F * v
            if Pm >= 0:
                Pb = Pm / ETA_DRIVE + P_IDLE
            elif F > -10.0:                                    # coasting, no regen
                Pb = P_IDLE
            else:
                Pb = max(Pm * ETA_REGEN, -P_REGEN_MAX) + P_IDLE
            Pb = min(Pb, P_MAX)

            # battery:  V = OCV(SoC) - I*R - V_rc ;  P = V*I
            R = r0 * (1 + 0.8 * (1 - soc) ** 2)
            vo = ocv(soc) - v1
            disc = max(vo * vo - 4 * R * Pb, 1.0)
            I = (vo - np.sqrt(disc)) / (2 * R)
            Vt = vo - I * R
            v1 += DT / TAU_RC * (I * R1 - v1)
            soc = np.clip(soc - I * DT / q_as, 0.0, 1.0)

            I_meas = I + i_off + rng.normal(0, 0.04)
            q_est_as -= I_meas * DT
            rows.append((trip_id, t * DT,
                         round(Vt + rng.normal(0, 0.015), 2),
                         round(I_meas, 2),
                         int(round(q_est_as / 3.6)),                 # mAh remaining (coulomb counting)
                         round(v * 3.6 + rng.normal(0, 0.2), 1),     # IMU speed (labeling only)
                         s))
            t += 1
    df = pd.DataFrame(rows, columns=["trip_id", "t", "voltage", "current", "capacity",
                                     "speed", "surface"])
    df["speeding"] = (df["speed"] > SPEED_LIMIT_KMH).astype(np.int8)
    df["sidewalk"] = (df["surface"] > 0).astype(np.int8)
    meta = dict(trip_id=trip_id, mass=round(mass, 1), temp=round(temp, 1), soh=round(soh, 3),
                r0=round(r0, 3), start_soc=round(float(np.clip(q_est_as / q_as, 0, 1)), 3))
    return df, meta


def random_trip_segments(rng, total_s=180):
    n = rng.integers(2, 5)
    cuts = np.sort(rng.uniform(25, total_s - 25, n - 1)) if n > 1 else []
    # enforce min length of 25 s
    while n > 1 and (np.diff(np.concatenate([[0], cuts, [total_s]])) < 25).any():
        cuts = np.sort(rng.uniform(25, total_s - 25, n - 1))
    durs = np.diff(np.concatenate([[0], cuts, [total_s]]))
    return [(float(d), int(rng.choice(14, p=SURF_P)), pick_speed(rng), pick_grade(rng)) for d in durs]


def scenario_segments(kind, rng):
    """100 s scenario, illegal behaviour in [20 s, 80 s] (paper Sec. 6.3)."""
    g = rng.uniform(-0.005, 0.005)
    legal_v = lambda: rng.uniform(8, 18)
    if kind == "speeding":
        return [(20, 0, legal_v(), g), (60, 0, rng.uniform(23.0, 24.5), g), (20, 0, legal_v(), g)]
    side = int(rng.integers(1, 14))
    return [(20, 0, legal_v(), g), (60, side, legal_v(), g), (20, 0, legal_v(), g)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=36.0)
    ap.add_argument("--trip_s", type=int, default=180)
    ap.add_argument("--scenarios", type=int, default=20, help="per behaviour")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    n_trips = int(round(a.hours * 3600 / a.trip_s))
    dfs, metas = [], []
    for k in range(n_trips):
        d, m = simulate(random_trip_segments(rng, a.trip_s), rng, k)
        dfs.append(d); metas.append(m)

    # group split by trip: 64 / 16 / 20
    perm = rng.permutation(n_trips)
    split = np.empty(n_trips, dtype=object)
    split[perm[:int(.64 * n_trips)]] = "train"
    split[perm[int(.64 * n_trips):int(.80 * n_trips)]] = "val"
    split[perm[int(.80 * n_trips):]] = "test"
    for m in metas: m["split"] = split[m["trip_id"]]

    tid = 100000
    for kind in ("speeding", "sidewalk"):
        for _ in range(a.scenarios):
            d, m = simulate(scenario_segments(kind, rng), rng, tid, calm=True)
            m["split"] = f"scenario_{kind}"
            dfs.append(d); metas.append(m); tid += 1

    meta = pd.DataFrame(metas)
    data = pd.concat(dfs, ignore_index=True)
    data = data.merge(meta[["trip_id", "split"]], on="trip_id")
    data.to_csv(f"{a.out}/trips.csv", index=False)
    meta.to_csv(f"{a.out}/trips_meta.csv", index=False)
    report(data, meta)


def report(data, meta):
    tr = data[~data.split.str.startswith("scenario")]
    print(f"\nrows={len(data):,}  trips={meta.shape[0]}  hours(train-type data)={len(tr)*DT/3600:.1f}")
    print("split trips:", meta.split.value_counts().to_dict())
    print(f"sidewalk share={tr.sidewalk.mean():.3f}  speeding share={tr.speeding.mean():.3f}")
    print("surface share % :", (tr.surface.value_counts(normalize=True).sort_index() * 100).round(2).to_dict())
    bins = pd.cut(tr.speed, [-1, 5, 10, 15, 20, 30])
    print("speed bin share % :", (bins.value_counts(normalize=True).sort_index() * 100).round(2).to_list())
    road = tr[tr.surface == 0]
    sb = pd.cut(road.speed, [-1, 10, 20, 30], labels=["<10", "10-20", ">20"])
    print("\nRoad voltage by speed (paper Fig 2a: med ~40.3 / 39.5 / 38.7 V):")
    print(road.groupby(sb, observed=True).voltage.agg(["mean", "median", "std"]).round(2))
    dv = tr.groupby("trip_id").voltage.diff()
    tr2 = tr.assign(dV=dv)
    print("\nstd of dV = V(t+1)-V(t) by surface (paper Fig 2b: sidewalk > road, varies by type):")
    print(tr2.groupby("surface").dV.std().round(3).to_dict())
    print(f"\nvoltage range {tr.voltage.min():.1f}..{tr.voltage.max():.1f} V | "
          f"current {tr.current.min():.1f}..{tr.current.max():.1f} A | "
          f"capacity {tr.capacity.min()}..{tr.capacity.max()} mAh")


if __name__ == "__main__":
    main()
