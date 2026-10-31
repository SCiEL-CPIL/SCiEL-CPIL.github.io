"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Hybrid Strategy:
    # For small items, use Worst Fit (maximize remaining space) to preserve tight fits for larger items.
    # For larger items, use Best Fit (minimize remaining space) to pack efficiently.
    
    # Threshold: 10% of capacity (100) = 10
    threshold = 10.0
    
    if item < threshold:
        # Worst Fit: Maximize remaining capacity after placement
        # Score is proportional to remaining space left
        scores = bins.copy()
    else:
        # Best Fit: Minimize remaining capacity after placement
        # Score is negative of remaining space left (so higher score = less remaining)
        scores = -(bins - item)
        
    # Mask invalid bins
    scores[~valid] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
