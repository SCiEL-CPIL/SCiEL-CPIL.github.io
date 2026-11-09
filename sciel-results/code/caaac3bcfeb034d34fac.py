"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    
    Uses a Worst-Fit strategy: prioritize bins with the largest remaining capacity.
    This avoids creating small, unusable fragments by keeping large spaces available
    for future items, and preserves smaller spaces for items that might only fit there.
    
    Returns an array of scores (higher is better) for bins with enough capacity.
    Bins with insufficient capacity get -inf.
    """
    remaining = bins - item
    
    # Initialize scores
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # Only consider bins with enough remaining capacity
    valid = remaining >= 0
    
    if not np.any(valid):
        # No valid bins; return all -inf
        return scores
    
    r = remaining[valid]
    
    # Worst Fit: Higher remaining capacity is better
    # This structurally differs from Best Fit (which minimizes remaining)
    # by preserving large gaps for larger items and avoiding fragmentation
    # of small gaps.
    scores[valid] = r
    
    return scores
# EVOLVE-BLOCK-END
