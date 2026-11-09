"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Best-fit with a small content bonus to reduce fragmentation
    # and a nonlinear penalty for large remaining spaces.
    remaining = bins - item
    # Only score bins that can fit the item
    valid = remaining >= 0
    # Nonlinear transformation: penalize large remaining capacity more
    # while adding a small bonus for bins that already have content
    score = np.where(valid, -(remaining ** 1.5) + 0.1 * bins, -np.inf)
    return score
# EVOLVE-BLOCK-END
