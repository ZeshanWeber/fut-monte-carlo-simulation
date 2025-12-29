# src/main.py
from __future__ import annotations

from pathlib import Path

from simulation import load_target_squad_csv, run_many_trials


PACK_PRICE_EUR = 10.0
N_TRIALS = 10_000
SEED = 42

CSV_PATH = Path("data/input/target_squad.csv")


def quantile(sorted_vals: list[int], q: float) -> float:
    """Simple quantile (no external libs)."""
    if not 0 <= q <= 1:
        raise ValueError("q must be within [0,1]")
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("empty list")
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main() -> None:
    players = load_target_squad_csv(CSV_PATH)

    results = run_many_trials(players, n_trials=N_TRIALS, seed=SEED)
    results_sorted = sorted(results)

    mean_packs = sum(results) / len(results)
    median_packs = quantile(results_sorted, 0.5)
    q95_packs = quantile(results_sorted, 0.95)

    print("=== FUT Monte Carlo (Model 1) ===")
    print(f"Trials: {N_TRIALS}")
    print(f"Pack price: {PACK_PRICE_EUR:.2f} EUR")
    print(f"Mean packs: {mean_packs:.2f}  -> Mean cost: {mean_packs * PACK_PRICE_EUR:.2f} EUR")
    print(f"Median packs: {median_packs:.2f} -> Median cost: {median_packs * PACK_PRICE_EUR:.2f} EUR")
    print(f"95% quantile packs: {q95_packs:.2f} -> 95% cost: {q95_packs * PACK_PRICE_EUR:.2f} EUR")


if __name__ == "__main__":
    main()

