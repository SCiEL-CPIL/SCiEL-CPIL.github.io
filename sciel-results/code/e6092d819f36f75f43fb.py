"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Best-fit with smooth penalty for very small remainders.
    # Prefer bins where (bins - item) is small but not too small.
    # Use a quadratic penalty on the remainder to smooth the objective.
    remaining = bins - item
    # Only consider bins with enough capacity
    valid = remaining >= 0
    # Score: minimize remaining capacity, but penalize extremely small remainders
    # that are likely useless for future items.
    # Use: score = -remaining + epsilon * remaining^2 / capacity^2
    # This makes the function prefer moderate remainders over near-zero ones.
    capacity = 100.0
    eps = 0.5  # weight of the quadratic term
    scores = -remaining + eps * (remaining / capacity) ** 2
    # Set invalid bins to very low score so they're never chosen
    scores = np.where(valid, scores, -np.inf)
    return scores
# EVOLVE-BLOCK-END
