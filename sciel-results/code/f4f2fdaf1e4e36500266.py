"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Smooth best-fit with power-law penalty on remaining capacity.
    # Higher power strengthens preference for tight fits (less remaining space),
    # which helps consolidate items into fewer bins (lower L1 excess).
    remaining = np.maximum(bins - item, 0.0)
    # Only score bins with enough capacity; others get -inf
    mask = bins >= item
    scores = np.full_like(bins, -np.inf)
    scores[mask] = -(remaining[mask] ** 1.5)
    return scores
# EVOLVE-BLOCK-END
