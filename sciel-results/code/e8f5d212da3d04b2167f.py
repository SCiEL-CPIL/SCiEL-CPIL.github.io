"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # First-Fit-Diminishing style: score by negative current remaining capacity.
    # This forces selection of the bin with the least available space that can
    # still fit the item, consolidating items and keeping other bins intact
    # for larger future arrivals.
    remaining = bins - item
    # For bins that cannot fit the item, remaining is negative; we want to
    # exclude them by giving them a very low score.
    # But the contract says "enough remaining capacity" — so we should only
    # score bins where bins >= item. For bins that can't fit, return -inf or
    # a very large negative so they are never selected.
    # Use np.where to set infeasible bins to a large negative value.
    feasible = bins >= item
    # For feasible bins, score = -remaining (i.e., prefer least remaining)
    # For infeasible bins, score = -inf (or very large negative)
    score = np.where(feasible, -remaining, -1e18)
    return score
# EVOLVE-BLOCK-END
