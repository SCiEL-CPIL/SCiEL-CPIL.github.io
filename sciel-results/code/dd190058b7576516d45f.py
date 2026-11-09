import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for an online bin-packing heuristic.
    
    Uses a Worst-Fit strategy: prefer bins with the most remaining capacity
    that can still fit the item. This preserves high-capacity bins for future
    large items and reduces fragmentation by spreading items across bins
    rather than filling them completely.
    """
    # Filter bins that can fit the item
    valid_mask = bins >= item
    
    # Initialize scores to a very low value for invalid bins
    scores = np.full_like(bins, -1e10, dtype=np.float64)
    
    if np.any(valid_mask):
        # For valid bins, score is the remaining capacity.
        # Maximizing this score selects the bin with the most free space (Worst-Fit).
        scores[valid_mask] = bins[valid_mask]
        
    return scores
