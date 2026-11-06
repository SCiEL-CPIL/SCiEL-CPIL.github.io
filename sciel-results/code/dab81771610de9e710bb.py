"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Combine best-fit (minimize remaining capacity) with a
    fragmentation penalty (prefer bins whose remaining capacity after
    placement is closer to the median of other feasible remainders).
    This smooths the selection landscape and reduces creation of
    disparate gaps that are hard to fill.
    
    For Weibull distributions, this should improve packing density
    compared to pure Best Fit.
    """
    # Track elapsed time for safety margin
    t0 = time.time()
    
    # Feasibility: bin must have enough remaining capacity
    feasible = bins >= item
    
    # Remaining capacity after placing item in each feasible bin
    remaining = bins - item
    # Only consider feasible bins
    rem_feasible = remaining[feasible]
    
    if len(rem_feasible) == 0:
        # No feasible bin; return zeros (caller should handle)
        return np.zeros_like(bins)
    
    # Best-fit score: prefer smallest remaining capacity (negative = higher score for smaller rem)
    best_fit_score = -rem_feasible
    
    # Fragmentation penalty: penalize bins whose remaining capacity is
    # far from the median of feasible remainders
    if len(rem_feasible) > 1:
        med = np.median(rem_feasible)
        frag_penalty = 0.1 * (rem_feasible - med) ** 2
    else:
        frag_penalty = np.zeros_like(rem_feasible)
    
    # Combined score: higher is better
    # We want to maximize, so we combine best-fit (which is negative for large rem)
    # and subtract fragmentation penalty
    scores_feasible = best_fit_score - frag_penalty
    
    # Build full score array: infeasible bins get very low score
    scores = np.full_like(bins, -1e18, dtype=np.float64)
    scores[feasible] = scores_feasible
    
    # Safety deadline guard: if we're taking too long, just return best-fit
    elapsed = time.time() - t0
    if elapsed > 0.5:  # 500ms safety margin per call
        # Fall back to simple best-fit
        return -(bins - item)
    
    return scores
# EVOLVE-BLOCK-END
