"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin for 150s limit

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing using a Best-Fit (First-Fit-Diminishing) heuristic.
    
    The score for each bin is the negative of its remaining capacity after placing the item.
    Bins that cannot fit the item are assigned a very low score to ensure they are never selected.
    """
    # Check deadline (though this function is fast, we comply with the directive)
    if time.time() > _DEADLINE:
        # Return a default score if we are about to timeout
        # This shouldn't happen for a single O(N) operation
        return np.zeros_like(bins, dtype=np.float64)

    remaining = bins - item
    feasible = remaining >= 0
    
    # Initialize scores to a very low value for infeasible bins
    scores = np.full_like(remaining, -1e18, dtype=np.float64)
    
    # For feasible bins, score is -remaining
    # This ensures the bin with the least remaining space gets the highest score (closest to 0)
    scores[feasible] = -remaining[feasible]
    
    return scores
# EVOLVE-BLOCK-END
