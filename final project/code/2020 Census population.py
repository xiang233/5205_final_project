#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apportionment Project (US House, 2020 Census)
- Deterministic methods
- Randomized quota rounding (systematic / sampford)
- Metrics
- Randomized summary (mean/std/p_change)
- Sensitivity:
    * deterministic: population noise -> seat flips vs baseline
    * randomized:
        (A) intrinsic randomness vs HH (population fixed)
        (B) population-noise sensitivity of expected seats (mean over inner draws)
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# -----------------------------
# Data loading
# -----------------------------
def load_us_2020_from_xlsx(xlsx_path: str) -> pd.DataFrame:
    df_raw = pd.read_excel(xlsx_path, engine="openpyxl")
    col0 = df_raw.columns[0]

    idx_list = df_raw.index[df_raw[col0].astype(str).str.strip().eq("STATE")].tolist()
    if not idx_list:
        raise ValueError("Could not find a header cell exactly 'STATE' in the first column.")
    start_idx = idx_list[0]

    df = df_raw.iloc[start_idx + 1 :].copy()
    cols = list(df.columns)
    if len(cols) < 3:
        raise ValueError("Unexpected table shape; need at least 3 columns after header row.")

    take = 4 if len(cols) >= 4 else 3
    df = df.iloc[:, :take].copy()

    if take == 4:
        df.columns = ["State", "Population", "HH_Seats_Official", "Misc"]
    else:
        df.columns = ["State", "Population", "HH_Seats_Official"]

    df["Population"] = pd.to_numeric(df["Population"], errors="coerce")
    df["HH_Seats_Official"] = pd.to_numeric(df["HH_Seats_Official"], errors="coerce")

    df = df[df["Population"].notna() & df["HH_Seats_Official"].notna()].copy()
    df = df[~df["State"].astype(str).str.contains("TOTAL", case=False, na=False)].copy()
    df = df[["State", "Population", "HH_Seats_Official"]].reset_index(drop=True)

    if df.shape[0] < 40:
        raise ValueError(f"Parsed too few rows ({df.shape[0]}). Check the xlsx format.")
    return df


# -----------------------------
# Deterministic apportionment
# -----------------------------
def apportion_hamilton(pop: np.ndarray, house_size: int, min_seat: int = 1) -> np.ndarray:
    n = len(pop)
    seats = np.full(n, min_seat, dtype=int)
    remaining = house_size - n * min_seat
    if remaining < 0:
        raise ValueError("house_size too small for min_seat constraint.")

    quotas = remaining * pop / pop.sum()
    base = np.floor(quotas).astype(int)
    seats += base

    left = remaining - int(base.sum())
    if left > 0:
        rema = quotas - np.floor(quotas)
        order = np.lexsort((-np.arange(n), -pop, -rema))
        for i in order[:left]:
            seats[i] += 1
    return seats


def apportion_priority(pop: np.ndarray, house_size: int, priority_fn, min_seat: int = 1) -> np.ndarray:
    n = len(pop)
    seats = np.full(n, min_seat, dtype=int)
    remaining = house_size - n * min_seat
    if remaining < 0:
        raise ValueError("house_size too small for min_seat constraint.")

    for _ in range(remaining):
        priorities = priority_fn(pop, seats)
        idx = int(np.argmax(priorities))
        seats[idx] += 1
    return seats


def apportion_jefferson(pop: np.ndarray, house_size: int, min_seat: int = 1) -> np.ndarray:
    return apportion_priority(pop, house_size, lambda P, s: P / (s + 1), min_seat=min_seat)


def apportion_webster(pop: np.ndarray, house_size: int, min_seat: int = 1) -> np.ndarray:
    return apportion_priority(pop, house_size, lambda P, s: P / (2 * s + 1), min_seat=min_seat)


def apportion_huntington_hill(pop: np.ndarray, house_size: int, min_seat: int = 1) -> np.ndarray:
    def hh(P, s):
        return P / np.sqrt(s * (s + 1))
    return apportion_priority(pop, house_size, hh, min_seat=min_seat)


# -----------------------------
# Randomized rounding
# -----------------------------
def systematic_rounding(p: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(p)
    if k == 0:
        return np.zeros(n, dtype=int)
    if not np.isclose(p.sum(), k, atol=1e-9):
        raise ValueError(f"systematic_rounding requires sum(p)=k; got sum={p.sum()}, k={k}")

    u = rng.uniform(0.0, 1.0)
    x = np.zeros(n, dtype=int)
    cum = 0.0
    for i in range(n):
        a = u + cum
        b = a + p[i]
        if math.ceil(a - 1e-12) < b - 1e-12:
            x[i] = 1
        cum += p[i]

    s = int(x.sum())
    if s != k:
        scores = p.copy()
        if s > k:
            drop = np.where(x == 1)[0]
            drop_order = drop[np.argsort(scores[drop])]
            x[drop_order[: (s - k)]] = 0
        else:
            add = np.where(x == 0)[0]
            add_order = add[np.argsort(-scores[add])]
            x[add_order[: (k - s)]] = 1
    return x


def sampford_rounding(p: np.ndarray, k: int, rng: np.random.Generator, max_tries: int = 2000) -> np.ndarray:
    n = len(p)
    if k == 0:
        return np.zeros(n, dtype=int)
    if np.any(p < -1e-12) or np.any(p > 1 - 1e-12):
        raise ValueError("Sampford rounding expects p in [0,1).")
    if not np.isclose(p.sum(), k, atol=1e-9):
        raise ValueError(f"sampford_rounding requires sum(p)=k; got sum={p.sum()}, k={k}")

    w1 = p.copy()
    if w1.sum() <= 0:
        return np.zeros(n, dtype=int)

    w2 = np.zeros(n, dtype=float)
    denom = 1.0 - p
    mask = denom > 1e-12
    w2[mask] = p[mask] / denom[mask]
    if w2.sum() <= 0:
        return systematic_rounding(p, k, rng)

    p1 = w1 / w1.sum()
    p2 = w2 / w2.sum()

    for _ in range(max_tries):
        i1 = int(rng.choice(n, p=p1))
        others = rng.choice(n, size=k - 1, replace=True, p=p2).tolist()
        draws = [i1] + others
        if len(set(draws)) == k:
            x = np.zeros(n, dtype=int)
            x[draws] = 1
            return x

    return systematic_rounding(p, k, rng)


def apportion_randomized_quota(
    pop: np.ndarray,
    house_size: int,
    rounding: str,
    rng: np.random.Generator,
    min_seat: int = 1,
) -> np.ndarray:
    n = len(pop)
    seats = np.full(n, min_seat, dtype=int)

    remaining = house_size - n * min_seat
    if remaining < 0:
        raise ValueError("house_size too small for min_seat constraint.")

    q = remaining * pop / pop.sum()
    base = np.floor(q).astype(int)
    seats += base

    residues = q - np.floor(q)
    k = remaining - int(base.sum())
    if k == 0:
        return seats

    s = float(residues.sum())
    if not np.isclose(s, k, atol=1e-9):
        residues = residues * (k / s)
        residues = np.clip(residues, 0.0, 1.0 - 1e-12)
        residues = residues * (k / float(residues.sum()))

    if rounding == "systematic":
        x = systematic_rounding(residues, k, rng)
    elif rounding == "sampford":
        x = sampford_rounding(residues, k, rng)
    else:
        raise ValueError("rounding must be 'systematic' or 'sampford'.")

    seats += x
    return seats


# -----------------------------
# Metrics
# -----------------------------
@dataclass
class Metrics:
    fairness_deviation: float
    max_ratio: float
    gini_like: float
    spearman_pop_vs_seat_err: float
    l1_quota_error: float


def compute_metrics(pop: np.ndarray, seats: np.ndarray, house_size: int, min_seat: int = 1) -> Metrics:
    total_pop = float(pop.sum())
    avg_people_per_seat = total_pop / house_size
    people_per_seat = pop / seats
    rho = people_per_seat / avg_people_per_seat

    fairness_deviation = float(np.sum(np.abs(rho - 1.0)))
    max_ratio = float(rho.max() / rho.min())

    x = np.sort(people_per_seat.astype(float))
    n = len(x)
    gini_like = float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum() + 1e-12))

    remaining = house_size - len(pop) * min_seat
    q = remaining * pop / pop.sum()
    extra = seats - min_seat
    l1_quota_error = float(np.sum(np.abs(extra - q)))

    pop_rank = pd.Series(pop).rank(method="average")
    seat_err = pd.Series(extra - q)
    spearman = float(pop_rank.corr(seat_err.rank(method="average")))

    return Metrics(
        fairness_deviation=fairness_deviation,
        max_ratio=max_ratio,
        gini_like=gini_like,
        spearman_pop_vs_seat_err=spearman,
        l1_quota_error=l1_quota_error,
    )


# -----------------------------
# Experiments
# -----------------------------
def sensitivity_experiment_deterministic(
    pop: np.ndarray,
    base_seats: np.ndarray,
    apportion_fn,
    house_size: int,
    rng: np.random.Generator,
    eps: float,
    R: int,
    desc: str,
) -> Dict[str, float]:
    """Deterministic: population noise -> flips vs baseline seats."""
    n = len(pop)
    total_flip = np.empty(R, dtype=float)
    any_change = 0

    for t in tqdm(range(R), desc=desc, leave=False):
        noise = rng.uniform(1.0 - eps, 1.0 + eps, size=n)
        pop2 = np.maximum(1.0, pop * noise)
        seats2 = apportion_fn(pop2, house_size)
        diff = np.abs(seats2 - base_seats)
        total_flip[t] = diff.sum() / 2.0
        if diff.sum() > 0:
            any_change += 1

    return {
        "avg_flips": float(total_flip.mean()),
        "std_flips": float(total_flip.std()),
        "p_any_change": float(any_change / R),
    }


def intrinsic_randomness_vs_reference(
    pop: np.ndarray,
    ref_seats: np.ndarray,
    house_size: int,
    rounding: str,
    rng: np.random.Generator,
    draws: int,
    min_seat: int,
    desc: str,
) -> Dict[str, float]:
    """Randomized (population fixed): draws -> flips vs a fixed reference (e.g., HH)."""
    n = len(pop)
    total_flip = np.empty(draws, dtype=float)
    any_change = 0

    for t in tqdm(range(draws), desc=desc, leave=False):
        seats = apportion_randomized_quota(pop, house_size, rounding=rounding, rng=rng, min_seat=min_seat)
        diff = np.abs(seats - ref_seats)
        total_flip[t] = diff.sum() / 2.0
        if diff.sum() > 0:
            any_change += 1

    return {
        "avg_flips": float(total_flip.mean()),
        "std_flips": float(total_flip.std()),
        "p_any_change": float(any_change / draws),
    }


def expected_seats(pop: np.ndarray, house_size: int, rounding: str, rng: np.random.Generator, draws: int, min_seat: int, desc: str) -> np.ndarray:
    """Estimate E[seats] by Monte Carlo mean."""
    acc = np.zeros(len(pop), dtype=float)
    for _ in tqdm(range(draws), desc=desc, leave=False):
        acc += apportion_randomized_quota(pop, house_size, rounding=rounding, rng=rng, min_seat=min_seat)
    return acc / draws


def sensitivity_randomized_expected(
    pop: np.ndarray,
    house_size: int,
    rounding: str,
    rng: np.random.Generator,
    eps: float,
    R: int,
    inner_draws: int,
    min_seat: int,
    desc: str,
) -> Dict[str, float]:
    """
    Randomized sensitivity to population noise, but comparing EXPECTED seats:
      - Compute baseline mu0 = E[seats | pop]
      - For each perturbation pop2: compute mu2 = E[seats | pop2]
      - metric: expected_flips = 0.5 * L1(mu2 - mu0)
    """
    mu0 = expected_seats(pop, house_size, rounding, rng, inner_draws, min_seat, desc=f"{desc} baseline E[seats]")

    flips = np.empty(R, dtype=float)
    any_change = 0
    for t in tqdm(range(R), desc=desc, leave=False):
        noise = rng.uniform(1.0 - eps, 1.0 + eps, size=len(pop))
        pop2 = np.maximum(1.0, pop * noise)
        mu2 = expected_seats(pop2, house_size, rounding, rng, inner_draws, min_seat, desc=f"{desc} inner")
        d = np.abs(mu2 - mu0)
        flips[t] = d.sum() / 2.0
        if d.sum() > 1e-9:
            any_change += 1

    return {
        "avg_expected_flips": float(flips.mean()),
        "std_expected_flips": float(flips.std()),
        "p_any_change": float(any_change / R),
    }


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def plot_seat_diff(states: List[str], pop: np.ndarray, seats_a: np.ndarray, seats_b: np.ndarray, title: str, outpath: str):
    diff = seats_a - seats_b
    x = pop.astype(float)

    plt.figure()
    plt.scatter(x, diff)
    plt.xscale("log")
    plt.xlabel("Population (log scale)")
    plt.ylabel("Seat difference")
    plt.title(title)
    plt.axhline(0.0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, default="apportionment-2020-table01.xlsx")
    parser.add_argument("--house_size", type=int, default=435)
    parser.add_argument("--min_seat", type=int, default=1)
    parser.add_argument("--eps", type=float, default=0.005, help="±eps noise (0.005 = 0.5%)")
    parser.add_argument("--R", type=int, default=2000, help="number of perturbation trials")
    parser.add_argument("--rand_draws", type=int, default=5000, help="draws for systematic randomized summary")
    parser.add_argument("--sampford_draws", type=int, default=300, help="draws for sampford summary (slow)")
    parser.add_argument("--intrinsic_draws", type=int, default=2000, help="draws for intrinsic randomness stats")
    parser.add_argument("--rand_inner", type=int, default=200, help="inner draws for expected seats under randomized method")
    parser.add_argument("--do_sampford_sensitivity", action="store_true",
                        help="also run Sampford expected-sensitivity (VERY slow)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ensure_dir(args.outdir)

    df = load_us_2020_from_xlsx(args.xlsx)
    states = df["State"].tolist()
    pop = df["Population"].to_numpy(dtype=float)
    hh_official = df["HH_Seats_Official"].to_numpy(dtype=int)

    # --- deterministic seats ---
    ham = apportion_hamilton(pop, args.house_size, min_seat=args.min_seat)
    jef = apportion_jefferson(pop, args.house_size, min_seat=args.min_seat)
    web = apportion_webster(pop, args.house_size, min_seat=args.min_seat)
    hh = apportion_huntington_hill(pop, args.house_size, min_seat=args.min_seat)

    # --- randomized (one draw for display) ---
    sys_one = apportion_randomized_quota(pop, args.house_size, rounding="systematic", rng=rng, min_seat=args.min_seat)
    samp_one = apportion_randomized_quota(pop, args.house_size, rounding="sampford", rng=rng, min_seat=args.min_seat)

    out = pd.DataFrame({
        "State": states,
        "Population": pop.astype(int),
        "HH_official": hh_official,
        "Hamilton": ham,
        "Jefferson": jef,
        "Webster": web,
        "HuntingtonHill": hh,
        "Rand_Systematic(one)": sys_one,
        "Rand_Sampford(one)": samp_one,
    })

    for m in ["Hamilton", "Jefferson", "Webster", "HuntingtonHill"]:
        out[m + "_diff_from_HHcomputed"] = out[m] - out["HuntingtonHill"]
    out["HHofficial_diff_from_HHcomputed"] = out["HH_official"] - out["HuntingtonHill"]

    out_csv = os.path.join(args.outdir, "seat_allocations.csv")
    out.to_csv(out_csv, index=False)

    print("\n=== Loaded states:", len(states), "| total pop:", int(pop.sum()), "===\n")
    print("Saved seat allocation table to:", out_csv)

    # --- Metrics ---
    methods = {
        "Hamilton": ham,
        "Jefferson": jef,
        "Webster": web,
        "HuntingtonHill": hh,
        "HH_official": hh_official,
        "Rand_Systematic(one)": sys_one,
        "Rand_Sampford(one)": samp_one,
    }

    metrics_rows = []
    for name, seats in methods.items():
        m = compute_metrics(pop, seats, args.house_size, min_seat=args.min_seat)
        metrics_rows.append({
            "method": name,
            "FD(sum|rho-1|)": m.fairness_deviation,
            "max_ratio(rho_max/rho_min)": m.max_ratio,
            "gini_like_people_per_seat": m.gini_like,
            "spearman(pop_rank, seat_err_rank)": m.spearman_pop_vs_seat_err,
            "L1_quota_error(extra vs quota)": m.l1_quota_error,
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("FD(sum|rho-1|)")
    metrics_path = os.path.join(args.outdir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print("Saved metrics to:", metrics_path)
    print("\n=== Metrics (sorted by FD) ===")
    print(metrics_df.to_string(index=False))

    # --- Randomized summary ---
    def draw_many(rounding: str, draws: int) -> pd.DataFrame:
        seats_mat = np.empty((draws, len(pop)), dtype=int)
        for t in tqdm(range(draws), desc=f"Randomized draws [{rounding}]", leave=False):
            seats_mat[t] = apportion_randomized_quota(
                pop, args.house_size, rounding=rounding, rng=rng, min_seat=args.min_seat
            )
        return pd.DataFrame({
            "State": states,
            f"{rounding}_mean": seats_mat.mean(axis=0),
            f"{rounding}_std": seats_mat.std(axis=0),
            f"{rounding}_p_change_from_HHcomputed": (seats_mat != hh[None, :]).mean(axis=0),
        })

    print(f"\n[Randomized summary] systematic draws = {args.rand_draws}")
    sys_stats = draw_many("systematic", args.rand_draws)

    print(f"[Randomized summary] sampford draws = {args.sampford_draws}")
    samp_stats = draw_many("sampford", args.sampford_draws)

    rand_stats = sys_stats.merge(samp_stats, on="State", how="left")
    rand_path = os.path.join(args.outdir, "randomized_summary.csv")
    rand_stats.to_csv(rand_path, index=False)
    print("Saved randomized draw summary to:", rand_path)

    # --- Sensitivity (fixed, interpretable) ---
    print("\n=== Sensitivity (±{:.3%}, R={}) ===".format(args.eps, args.R))
    sens_rows = []

    det = {
        "Hamilton": (ham, lambda P, S: apportion_hamilton(P, S, min_seat=args.min_seat)),
        "Jefferson": (jef, lambda P, S: apportion_jefferson(P, S, min_seat=args.min_seat)),
        "Webster": (web, lambda P, S: apportion_webster(P, S, min_seat=args.min_seat)),
        "HuntingtonHill": (hh, lambda P, S: apportion_huntington_hill(P, S, min_seat=args.min_seat)),
    }

    for name, (base_seats, fn) in det.items():
        s = sensitivity_experiment_deterministic(
            pop, base_seats, fn, args.house_size, rng, args.eps, args.R, desc=f"Sensitivity [{name}]"
        )
        sens_rows.append({"method": name, **s})

    # Randomized: (A) intrinsic randomness vs HH
    s_intr = intrinsic_randomness_vs_reference(
        pop, hh, args.house_size, rounding="systematic", rng=rng, draws=args.intrinsic_draws,
        min_seat=args.min_seat, desc="Intrinsic randomness [systematic vs HH]"
    )
    sens_rows.append({"method": "Rand_systematic_intrinsic_vsHH", **s_intr})

    # Randomized: (B) noise sensitivity of EXPECTED seats
    s_exp = sensitivity_randomized_expected(
        pop, args.house_size, rounding="systematic", rng=rng, eps=args.eps, R=args.R,
        inner_draws=args.rand_inner, min_seat=args.min_seat, desc="Sensitivity expected [systematic]"
    )
    sens_rows.append({"method": "Rand_systematic_noise_on_mean", **s_exp})

    if args.do_sampford_sensitivity:
        s_intr2 = intrinsic_randomness_vs_reference(
            pop, hh, args.house_size, rounding="sampford", rng=rng, draws=min(args.intrinsic_draws, 500),
            min_seat=args.min_seat, desc="Intrinsic randomness [sampford vs HH]"
        )
        sens_rows.append({"method": "Rand_sampford_intrinsic_vsHH", **s_intr2})

        s_exp2 = sensitivity_randomized_expected(
            pop, args.house_size, rounding="sampford", rng=rng, eps=args.eps, R=min(args.R, 400),
            inner_draws=min(args.rand_inner, 100), min_seat=args.min_seat, desc="Sensitivity expected [sampford]"
        )
        sens_rows.append({"method": "Rand_sampford_noise_on_mean", **s_exp2})
    else:
        print("[Sensitivity] Skipping Sampford expected-sensitivity (use --do_sampford_sensitivity to enable)")

    sens_df = pd.DataFrame(sens_rows)
    sens_path = os.path.join(args.outdir, "sensitivity.csv")
    sens_df.to_csv(sens_path, index=False)
    print(sens_df.to_string(index=False))
    print("Saved sensitivity summary to:", sens_path)

    # --- Plots ---
    plot_seat_diff(states, pop, ham, hh, "Hamilton - HuntingtonHill (computed)", os.path.join(args.outdir, "diff_hamilton_vs_hh.png"))
    plot_seat_diff(states, pop, jef, hh, "Jefferson - HuntingtonHill (computed)", os.path.join(args.outdir, "diff_jefferson_vs_hh.png"))
    plot_seat_diff(states, pop, web, hh, "Webster - HuntingtonHill (computed)", os.path.join(args.outdir, "diff_webster_vs_hh.png"))
    plot_seat_diff(states, pop, hh_official, hh, "HH official - HH (computed)", os.path.join(args.outdir, "diff_hhofficial_vs_hhcomputed.png"))

    print("\nAll done. Outputs are in:", args.outdir)


if __name__ == "__main__":
    main()