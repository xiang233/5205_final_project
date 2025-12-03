#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apportionment Project (US House, 2020 Census)
- Deterministic methods: Hamilton, Jefferson, Webster, Huntington–Hill (HH)
- Randomized quota rounding: systematic / sampford
- Metrics
- Randomized summary (mean/std/p_change)
- Sensitivity:
    * deterministic: population noise -> seat flips vs baseline
    * randomized:
        (A) intrinsic randomness vs HH (population fixed)
        (B) population-noise sensitivity of expected seats (mean over inner draws)

Plus: PPT-friendly plots (PDF+PNG, 16:9-ish) saved to outputs/figs/

Usage example:
  python apportionment_project.py --xlsx apportionment-2020-table01.xlsx --eps 0.005 --R 2000 --seed 0
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# -----------------------------
# RNG helpers (avoid coupling inner/outer loops)
# -----------------------------
def make_rng(base_seed: int, *keys: int) -> np.random.Generator:
    ss = np.random.SeedSequence([base_seed, *keys])
    return np.random.default_rng(ss)


# -----------------------------
# Data loading
# -----------------------------
def load_us_2020_from_xlsx(xlsx_path: str) -> pd.DataFrame:
    """
    Parse the table where a row contains 'STATE' in the first column.
    After that row, data rows contain: State | Population | Number of Representatives (HH official) | ...
    Returns df columns: State, Population, HH_Seats_Official
    """
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
        # tie-break: larger remainder, then larger population, then lower index
        order = np.lexsort((np.arange(n), -pop, -rema))
        for i in order[:left]:
            seats[i] += 1
    return seats


def apportion_priority(pop: np.ndarray, house_size: int, priority_fn, min_seat: int = 1) -> np.ndarray:
    n = len(pop)
    seats = np.full(n, min_seat, dtype=int)
    remaining = house_size - n * min_seat
    if remaining < 0:
        raise ValueError("house_size too small for min_seat constraint.")

    # priority iteration (435 is small; O(n*435) is fine)
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
    """
    Systematic sampling / Grimmett rounding.
    Input: p in [0,1), sum(p)=k (integer).
    Output: x in {0,1}^n, sum(x)=k, E[x_i]=p_i.
    """
    n = len(p)
    if k == 0:
        return np.zeros(n, dtype=int)
    if not np.isclose(p.sum(), k, atol=1e-9):
        raise ValueError(f"systematic_rounding requires sum(p)=k; got sum={p.sum()}, k={k}")

    u = float(rng.uniform(0.0, 1.0))
    x = np.zeros(n, dtype=int)
    cum = 0.0
    for i in range(n):
        a = u + cum
        b = a + float(p[i])
        # integer exists in [a, b) iff ceil(a) < b
        if math.ceil(a - 1e-12) < b - 1e-12:
            x[i] = 1
        cum += float(p[i])

    # Numerical guard: enforce exact k
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
    """
    Sampford rounding (rejective procedure; can be slow when k is large-ish).
    Input: p in [0,1), sum(p)=k
    Output: x in {0,1}^n, sum(x)=k (falls back to systematic on too many rejections).
    """
    n = len(p)
    if k == 0:
        return np.zeros(n, dtype=int)
    if np.any(p < -1e-12) or np.any(p > 1 - 1e-9):
        raise ValueError("Sampford rounding expects p in [0,1).")
    if not np.isclose(p.sum(), k, atol=1e-9):
        raise ValueError(f"sampford_rounding requires sum(p)=k; got sum={p.sum()}, k={k}")

    w1 = p.copy()
    if w1.sum() <= 0:
        return np.zeros(n, dtype=int)

    denom = 1.0 - p
    w2 = np.zeros(n, dtype=float)
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

    # fallback
    return systematic_rounding(p, k, rng)


def apportion_randomized_quota(
    pop: np.ndarray,
    house_size: int,
    rounding: str,
    rng: np.random.Generator,
    min_seat: int = 1,
) -> np.ndarray:
    """
    Quota-based randomized apportionment:
      1) give min_seat to all
      2) compute remaining quotas q
      3) give floors
      4) distribute leftover seats by randomized rounding of residues
    """
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

    # renormalize residues to sum exactly k (float guard)
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
    eps: float,
    R: int,
    seed: int,
    desc: str,
    return_samples: bool = True,
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    """
    Deterministic sensitivity: population noise -> flips vs baseline seats.
    flips := 0.5*L1(seats2 - base_seats) (integer-valued in deterministic case)
    """
    n = len(pop)
    rng = make_rng(seed, 111, hash(desc) % (2**31 - 1))
    flips = np.empty(R, dtype=float)
    any_change = 0

    for t in tqdm(range(R), desc=desc, leave=False):
        noise = rng.uniform(1.0 - eps, 1.0 + eps, size=n)
        pop2 = np.maximum(1.0, pop * noise)
        seats2 = apportion_fn(pop2, house_size)
        diff = np.abs(seats2 - base_seats)
        flips[t] = diff.sum() / 2.0
        if diff.sum() > 0:
            any_change += 1

    stats = {
        "avg_flips": float(flips.mean()),
        "std_flips": float(flips.std()),
        "p_any_change": float(any_change / R),
    }
    return stats, (flips if return_samples else None)


def intrinsic_randomness_vs_reference(
    pop: np.ndarray,
    ref_seats: np.ndarray,
    house_size: int,
    rounding: str,
    draws: int,
    min_seat: int,
    seed: int,
    desc: str,
    return_samples: bool = True,
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    """
    Randomized (population fixed): draws -> flips vs a fixed reference (e.g., HH seats).
    """
    n = len(pop)
    rng = make_rng(seed, 222, hash(desc) % (2**31 - 1))
    flips = np.empty(draws, dtype=float)
    any_change = 0

    for t in tqdm(range(draws), desc=desc, leave=False):
        seats = apportion_randomized_quota(pop, house_size, rounding=rounding, rng=rng, min_seat=min_seat)
        diff = np.abs(seats - ref_seats)
        flips[t] = diff.sum() / 2.0
        if diff.sum() > 0:
            any_change += 1

    stats = {
        "avg_flips": float(flips.mean()),
        "std_flips": float(flips.std()),
        "p_any_change": float(any_change / draws),
    }
    return stats, (flips if return_samples else None)


def expected_seats(
    pop: np.ndarray,
    house_size: int,
    rounding: str,
    draws: int,
    min_seat: int,
    seed: int,
    desc: str,
) -> np.ndarray:
    """Estimate E[seats] by Monte Carlo mean."""
    rng = make_rng(seed, 333, hash(desc) % (2**31 - 1))
    acc = np.zeros(len(pop), dtype=float)
    for _ in tqdm(range(draws), desc=desc, leave=False):
        acc += apportion_randomized_quota(pop, house_size, rounding=rounding, rng=rng, min_seat=min_seat)
    return acc / draws


def sensitivity_randomized_expected(
    pop: np.ndarray,
    house_size: int,
    rounding: str,
    eps: float,
    R: int,
    inner_draws: int,
    min_seat: int,
    seed: int,
    desc: str,
    return_samples: bool = True,
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    """
    Randomized sensitivity to population noise, comparing EXPECTED seats:
      - baseline mu0 = E[seats | pop]
      - for each perturbation pop2: mu2 = E[seats | pop2]
      - expected_flips = 0.5 * L1(mu2 - mu0)
    """
    mu0 = expected_seats(
        pop, house_size, rounding,
        draws=inner_draws, min_seat=min_seat,
        seed=seed, desc=f"{desc} baseline E[seats]"
    )

    outer_rng = make_rng(seed, 444, hash(desc) % (2**31 - 1))
    flips = np.empty(R, dtype=float)
    any_change = 0

    for t in tqdm(range(R), desc=desc, leave=False):
        noise = outer_rng.uniform(1.0 - eps, 1.0 + eps, size=len(pop))
        pop2 = np.maximum(1.0, pop * noise)
        mu2 = expected_seats(
            pop2, house_size, rounding,
            draws=inner_draws, min_seat=min_seat,
            seed=seed + 100000 + t,  # change seed per outer trial
            desc=f"{desc} inner[{t}]"
        )
        d = np.abs(mu2 - mu0)
        flips[t] = d.sum() / 2.0
        if d.sum() > 1e-9:
            any_change += 1

    stats = {
        "avg_expected_flips": float(flips.mean()),
        "std_expected_flips": float(flips.std()),
        "p_any_change": float(any_change / R),
    }
    return stats, (flips if return_samples else None)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# -----------------------------
# PPT-friendly plots
# -----------------------------
def _save_fig(fig, outdir: str, name: str):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_diff_barh(states, diff, title, outdir, name, topk=20):
    idx = np.argsort(-np.abs(diff))[:topk]
    st = np.array(states)[idx]
    d = np.array(diff)[idx]

    fig = plt.figure(figsize=(10, 5.625))  # 16:9
    ax = fig.add_subplot(111)
    y = np.arange(len(st))
    ax.barh(y, d)
    ax.set_yticks(y)
    ax.set_yticklabels(st)
    ax.invert_yaxis()
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Seat difference (method - HH)")
    ax.set_title(title)
    _save_fig(fig, outdir, name)


def plot_diff_heatmap(states, pop, diff_matrix, method_names, outdir, name):
    order = np.argsort(pop)  # small -> large
    mat = diff_matrix[order, :]
    st = np.array(states)[order]

    fig = plt.figure(figsize=(11, 6.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(method_names)))
    ax.set_xticklabels(method_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(st)))
    ax.set_yticklabels(st, fontsize=7)
    ax.set_title("Seat differences vs HH (states sorted by population)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Seats (method - HH)")
    _save_fig(fig, outdir, name)


def plot_rho_boxplot(pop, seats_dict, house_size, outdir, name):
    avg = pop.sum() / house_size

    labels = []
    data = []
    for m, seats in seats_dict.items():
        rho = (pop / seats) / avg
        labels.append(m)
        data.append(rho)

    fig = plt.figure(figsize=(10, 5.625))
    ax = fig.add_subplot(111)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_ylabel("rho = (people/seat) / (national avg)")
    ax.set_title("Representation ratio distribution (closer to 1 is fairer)")
    plt.xticks(rotation=20, ha="right")
    _save_fig(fig, outdir, name)


def plot_pop_vs_seat_error(pop, seats, house_size, min_seat, title, outdir, name):
    remaining = house_size - len(pop) * min_seat
    q = remaining * pop / pop.sum()
    extra = seats - min_seat
    err = extra - q

    fig = plt.figure(figsize=(10, 5.625))
    ax = fig.add_subplot(111)
    ax.scatter(pop, err)
    ax.set_xscale("log")
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Population (log scale)")
    ax.set_ylabel("Seat error: (extra seats - quota)")
    ax.set_title(title)
    _save_fig(fig, outdir, name)


def plot_sensitivity_hist(flips_by_method: dict, outdir, name, bins=25):
    fig = plt.figure(figsize=(10, 5.625))
    ax = fig.add_subplot(111)
    for m, flips in flips_by_method.items():
        if flips is None:
            continue
        ax.hist(flips, bins=bins, alpha=0.5, label=m)
    ax.set_xlabel("Seat flips (L1/2)")
    ax.set_ylabel("Count")
    ax.set_title("Sensitivity: distribution of seat flips under population perturbation")
    ax.legend()
    _save_fig(fig, outdir, name)


def plot_randomized_pchange(states, p_change, outdir, name, topk=20, title="Randomized: P(seats != HH) by state"):
    idx = np.argsort(-p_change)[:topk]
    st = np.array(states)[idx]
    pc = np.array(p_change)[idx]

    fig = plt.figure(figsize=(10, 5.625))
    ax = fig.add_subplot(111)
    y = np.arange(len(st))
    ax.barh(y, pc)
    ax.set_yticks(y)
    ax.set_yticklabels(st)
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_title(title)
    _save_fig(fig, outdir, name)


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
                        help="also run Sampford intrinsic + expected sensitivity (VERY slow)")
    parser.add_argument("--save_flip_samples", action="store_true",
                        help="save flip arrays to outputs/flip_samples.npz (helpful for later plotting)")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    figdir = os.path.join(args.outdir, "figs")
    ensure_dir(figdir)

    df = load_us_2020_from_xlsx(args.xlsx)
    states = df["State"].tolist()
    pop = df["Population"].to_numpy(dtype=float)
    hh_official = df["HH_Seats_Official"].to_numpy(dtype=int)

    print("\n=== Loaded states:", len(states), "| total pop:", int(pop.sum()), "===\n")

    # --- deterministic seats ---
    ham = apportion_hamilton(pop, args.house_size, min_seat=args.min_seat)
    jef = apportion_jefferson(pop, args.house_size, min_seat=args.min_seat)
    web = apportion_webster(pop, args.house_size, min_seat=args.min_seat)
    hh = apportion_huntington_hill(pop, args.house_size, min_seat=args.min_seat)

    # --- randomized (one draw for display) ---
    rng_one = make_rng(args.seed, 999)
    sys_one = apportion_randomized_quota(pop, args.house_size, rounding="systematic", rng=rng_one, min_seat=args.min_seat)
    samp_one = apportion_randomized_quota(pop, args.house_size, rounding="sampford", rng=rng_one, min_seat=args.min_seat)

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
    print("Saved seat allocation table to:", out_csv)

    # --- Metrics ---
    methods = {
        "HuntingtonHill": hh,
        "HH_official": hh_official,
        "Webster": web,
        "Jefferson": jef,
        "Hamilton": ham,
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

    # --- Randomized summary (mean/std/p_change) ---
    def draw_many(rounding: str, draws: int, seed_key: int) -> pd.DataFrame:
        seats_mat = np.empty((draws, len(pop)), dtype=int)
        rng = make_rng(args.seed, 12345, seed_key)
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
    sys_stats = draw_many("systematic", args.rand_draws, seed_key=1)

    print(f"[Randomized summary] sampford draws = {args.sampford_draws}")
    samp_stats = draw_many("sampford", args.sampford_draws, seed_key=2)

    rand_stats = sys_stats.merge(samp_stats, on="State", how="left")
    rand_path = os.path.join(args.outdir, "randomized_summary.csv")
    rand_stats.to_csv(rand_path, index=False)
    print("Saved randomized draw summary to:", rand_path)

    # --- Sensitivity ---
    print("\n=== Sensitivity (±{:.3%}, R={}) ===".format(args.eps, args.R))
    sens_rows = []
    flip_samples = {}

    det = {
        "Hamilton": (ham, lambda P, S: apportion_hamilton(P, S, min_seat=args.min_seat)),
        "Jefferson": (jef, lambda P, S: apportion_jefferson(P, S, min_seat=args.min_seat)),
        "Webster": (web, lambda P, S: apportion_webster(P, S, min_seat=args.min_seat)),
        "HuntingtonHill": (hh, lambda P, S: apportion_huntington_hill(P, S, min_seat=args.min_seat)),
    }

    for name, (base_seats, fn) in det.items():
        s, flips = sensitivity_experiment_deterministic(
            pop, base_seats, fn, args.house_size, args.eps, args.R, seed=args.seed,
            desc=f"Sensitivity [{name}]", return_samples=True
        )
        sens_rows.append({"method": name, **s})
        flip_samples[f"det_{name}"] = flips

    # Randomized: (A) intrinsic randomness vs HH
    s_intr, flips_intr = intrinsic_randomness_vs_reference(
        pop, hh, args.house_size, rounding="systematic",
        draws=args.intrinsic_draws, min_seat=args.min_seat, seed=args.seed,
        desc="Intrinsic randomness [systematic vs HH]", return_samples=True,
    )
    sens_rows.append({"method": "Rand_systematic_intrinsic_vsHH", **s_intr})
    flip_samples["rand_systematic_intrinsic_vsHH"] = flips_intr

    # Randomized: (B) noise sensitivity of EXPECTED seats
    s_exp, flips_exp = sensitivity_randomized_expected(
        pop, args.house_size, rounding="systematic",
        eps=args.eps, R=args.R, inner_draws=args.rand_inner,
        min_seat=args.min_seat, seed=args.seed,
        desc="Sensitivity expected [systematic]", return_samples=True,
    )
    sens_rows.append({"method": "Rand_systematic_noise_on_mean", **s_exp})
    flip_samples["rand_systematic_noise_on_mean"] = flips_exp

    if args.do_sampford_sensitivity:
        s_intr2, flips_intr2 = intrinsic_randomness_vs_reference(
            pop, hh, args.house_size, rounding="sampford",
            draws=min(args.intrinsic_draws, 500), min_seat=args.min_seat, seed=args.seed,
            desc="Intrinsic randomness [sampford vs HH]", return_samples=True,
        )
        sens_rows.append({"method": "Rand_sampford_intrinsic_vsHH", **s_intr2})
        flip_samples["rand_sampford_intrinsic_vsHH"] = flips_intr2

        s_exp2, flips_exp2 = sensitivity_randomized_expected(
            pop, args.house_size, rounding="sampford",
            eps=args.eps, R=min(args.R, 400), inner_draws=min(args.rand_inner, 100),
            min_seat=args.min_seat, seed=args.seed,
            desc="Sensitivity expected [sampford]", return_samples=True,
        )
        sens_rows.append({"method": "Rand_sampford_noise_on_mean", **s_exp2})
        flip_samples["rand_sampford_noise_on_mean"] = flips_exp2
    else:
        print("[Sensitivity] Skipping Sampford sensitivity (use --do_sampford_sensitivity to enable)")

    sens_df = pd.DataFrame(sens_rows)
    sens_path = os.path.join(args.outdir, "sensitivity.csv")
    sens_df.to_csv(sens_path, index=False)
    print(sens_df.to_string(index=False))
    print("Saved sensitivity summary to:", sens_path)

    if args.save_flip_samples:
        npz_path = os.path.join(args.outdir, "flip_samples.npz")
        np.savez(npz_path, **{k: v for k, v in flip_samples.items() if v is not None})
        print("Saved flip samples to:", npz_path)

    # -----------------------------
    # Plots for analysis + PPT
    # -----------------------------
    # 1) Top seat differences vs HH (barh)
    plot_diff_barh(states, ham - hh, "Hamilton vs HH (top differences)", figdir, "bar_hamilton_vs_hh", topk=20)
    plot_diff_barh(states, jef - hh, "Jefferson vs HH (top differences)", figdir, "bar_jefferson_vs_hh", topk=20)
    plot_diff_barh(states, web - hh, "Webster vs HH (top differences)", figdir, "bar_webster_vs_hh", topk=20)
    plot_diff_barh(states, hh_official - hh, "HH official vs HH (sanity check)", figdir, "bar_hhofficial_vs_hh", topk=20)

    # 2) Heatmap of diffs (rows sorted by pop)
    diff_mat = np.stack([ham - hh, jef - hh, web - hh], axis=1)
    plot_diff_heatmap(states, pop, diff_mat, ["Hamilton", "Jefferson", "Webster"], figdir, "heatmap_diffs_vs_hh")

    # 3) rho distribution boxplot
    plot_rho_boxplot(
        pop,
        {"HH": hh, "Webster": web, "Jefferson": jef, "Hamilton": ham},
        args.house_size,
        figdir,
        "box_rho_methods",
    )

    # 4) size bias scatter: pop vs seat error (extra - quota)
    plot_pop_vs_seat_error(pop, ham, args.house_size, args.min_seat, "Hamilton: seat error vs population", figdir, "scatter_err_hamilton")
    plot_pop_vs_seat_error(pop, jef, args.house_size, args.min_seat, "Jefferson: seat error vs population", figdir, "scatter_err_jefferson")
    plot_pop_vs_seat_error(pop, web, args.house_size, args.min_seat, "Webster: seat error vs population", figdir, "scatter_err_webster")

    # 5) sensitivity flip histogram (deterministic)
    plot_sensitivity_hist(
        {
            "Hamilton": flip_samples.get("det_Hamilton"),
            "Jefferson": flip_samples.get("det_Jefferson"),
            "Webster": flip_samples.get("det_Webster"),
            "HH": flip_samples.get("det_HuntingtonHill"),
        },
        figdir,
        "hist_sensitivity_deterministic",
        bins=25,
    )

    # 6) randomized: top states by P(change from HH)
    if "systematic_p_change_from_HHcomputed" in rand_stats.columns:
        plot_randomized_pchange(
            states,
            rand_stats["systematic_p_change_from_HHcomputed"].to_numpy(),
            figdir,
            "bar_rand_systematic_pchange_top20",
            topk=20,
            title="Systematic randomized: P(seats != HH) (top 20 states)",
        )

    print("\nAll done. Outputs are in:", args.outdir)
    print("Figures for PPT/report are in:", figdir)


if __name__ == "__main__":
    main()