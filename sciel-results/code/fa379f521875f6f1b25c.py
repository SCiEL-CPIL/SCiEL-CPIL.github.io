# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item is placed in the bin with the highest score.
    
    Strategy: Worst Fit.
    We choose the bin that has the most remaining space after placing the item.
    This helps to balance the load across bins and avoid creating many bins
    with small, unusable remainders. It is a distinct approach from Best Fit
    (which minimizes remaining space).
    """
    remaining = bins - item
    
    # Create a score array, initialize to -inf for infeasible bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # For feasible bins (remaining >= 0), compute score
    feasible = remaining >= 0
    if np.any(feasible):
        # Worst Fit: Maximize the remaining space
        scores[feasible] = remaining[feasible]
        
    return scores
# EVOLVE-BLOCK-END
