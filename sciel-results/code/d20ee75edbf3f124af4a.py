"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    Higher score = preferred bin.
    
    Strategy: Modified best-fit with smooth surrogate.
    We use score = -sqrt(remaining) which:
    - Prefers bins with smaller remaining capacity (best-fit behavior)
    - Has diminishing sensitivity for large remaining (less aggressive than linear)
    - Is smooth (differentiable) which may help with optimization
    
    For Weibull distributions with capacity 100, this balances:
    1. Filling bins as much as possible (minimize number of bins used)
    2. Avoiding creating tiny unusable fragments
    """
    t0 = time.time()
    
    # Bins that can fit the item
    remaining = bins - item
    valid = remaining >= 0
    
    # Compute scores using smooth surrogate: -sqrt(rem + epsilon)
    # This is monotonically decreasing in remaining, preferring smaller remainders
    # The sqrt makes the preference less aggressive for large remainders
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    if np.any(valid):
        rem = remaining[valid]
        # Add small epsilon to avoid issues at rem=0
        scores[valid] = -np.sqrt(rem + 1e-9)
    
    # Deadline guard - keep simple, this is O(n) so should be fast
    if time.time() - t0 > 110:
        # Fallback: pure best-fit
        scores = np.where(valid, -remaining, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
