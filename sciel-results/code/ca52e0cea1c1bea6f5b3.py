"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Pure Best-Fit.
    Place the item in the bin with the smallest remaining capacity that can fit it.
    This minimizes the waste in the chosen bin, which generally leads to better
    consolidation and fewer total bins used compared to First-Fit or Worst-Fit.
    
    The current program's "enhanced" approach with consolidation bias was shown
    to be equivalent to best-fit (linearly dependent terms) and scored slightly
    worse (0.9998 vs 1.0000). We revert to the pure best-fit strategy which
    has been empirically confirmed as the best performer in the inspirations.
    
    Score = -(bins - item) for feasible bins (bins[i] >= item)
           = -inf for infeasible bins
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Emergency: return a simple fallback
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        return np.full(len(bins), -np.inf)
    
    scores = np.full(len(bins), -np.inf)
    scores[feasible] = -remaining_after[feasible]
    
    return scores
# EVOLVE-BLOCK-END
