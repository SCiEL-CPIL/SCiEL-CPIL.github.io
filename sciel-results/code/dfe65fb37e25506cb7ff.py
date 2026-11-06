# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    
    Uses a Best-Fit (First-Fit-Diminishing inspired) heuristic:
    - Selects the bin with the smallest remaining capacity after placing the item.
    - This is equivalent to selecting the bin with the smallest current capacity among valid bins.
    - Aggressively consolidates items to keep larger bins empty for larger future arrivals.
    """
    # Ensure bins is float for consistent arithmetic
    bins_float = bins.astype(np.float64, copy=False)
    
    remaining = bins_float - item
    valid = remaining >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full(bins_float.shape, -np.inf, dtype=np.float64)
    
    if np.any(valid):
        # Score is negative remaining capacity.
        # Higher score (closer to 0) means smaller remaining capacity (tighter fit).
        scores[valid] = -remaining[valid]
        
    return scores
# EVOLVE-BLOCK-END
