# src/simulation.py
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class TargetPlayer:
    """A target player and its probability to be drawn in one pack (Model 1)."""
    name: str
    p: float


def load_target_squad_csv(path: str | Path) -> List[TargetPlayer]:
    """
    Loads target players from a CSV file with columns:
    player_name,p
    """
    path = Path(path)
    players: List[TargetPlayer] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["player_name"].strip()
            p = float(row["p"])
            players.append(TargetPlayer(name=name, p=p))

    if not players:
        raise ValueError("CSV contains no players.")

    return players


def _build_distribution(players: List[TargetPlayer]) -> Tuple[List[str], List[float]]:
    """
    Builds a categorical distribution over: [player names..., 'OTHER'].
    'OTHER' means: a non-target player was drawn.
    """
    if any(pl.p < 0 for pl in players):
        raise ValueError("Probabilities must be non-negative.")

    p_hit = sum(pl.p for pl in players)
    if p_hit > 1.0:
        raise ValueError(f"Sum of target probabilities exceeds 1.0: {p_hit}")

    outcomes = [pl.name for pl in players] + ["OTHER"]
    probs = [pl.p for pl in players] + [1.0 - p_hit]

    # Normalize (small numeric safeguard)
    s = sum(probs)
    probs = [p / s for p in probs]
    return outcomes, probs


def run_single_trial(players: List[TargetPlayer], rng: random.Random | None = None) -> int:
    """
    Runs one Monte Carlo trial.
    Model 1: each pack produces exactly one draw event (target player or OTHER).
    Returns the number of packs needed to collect all target players at least once.
    """
    rng = rng or random.Random()
    outcomes, probs = _build_distribution(players)

    target_set = {pl.name for pl in players}
    collected = set()

    packs_opened = 0
    while collected != target_set:
        packs_opened += 1
        draw = rng.choices(outcomes, weights=probs, k=1)[0]
        if draw != "OTHER":
            collected.add(draw)

    return packs_opened


def run_many_trials(players: List[TargetPlayer], n_trials: int, seed: int = 42) -> List[int]:
    """
    Runs many independent trials and returns a list of packs_needed.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")

    rng = random.Random(seed)
    return [run_single_trial(players, rng=rng) for _ in range(n_trials)]
