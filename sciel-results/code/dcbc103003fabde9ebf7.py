import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Uses a Best-Fit Decreasing heuristic variant with smooth scoring.
    We want to minimize the remaining space in the chosen bin to pack tightly.
    Score = -remaining_capacity (lower remaining is better -> higher score).
    
    To handle the constraint that bins must have enough capacity:
    We assign a very low score (e.g., -inf or a large negative number) to bins 
    that cannot fit the item, ensuring they are never selected.
    
    For valid bins, we prefer the one with the least remaining capacity after placement.
    Score = - (bins - item)
    """
    remaining = bins - item
    valid_mask = remaining >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if np.any(valid_mask):
        # For valid bins, score is negative of the remaining capacity after insertion.
        # This effectively implements Best Fit: the bin that leaves the least space.
        scores[valid_mask] = -remaining[valid_mask]
    
    return scores
