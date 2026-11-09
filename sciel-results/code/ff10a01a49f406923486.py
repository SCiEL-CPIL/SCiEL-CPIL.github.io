import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Use Worst-Fit strategy: prefer bins with the most remaining capacity
    # among those that can fit the item.
    
    # Initialize with a very low score for invalid bins
    scores = np.full(len(bins), -1e10)
    
    # Identify valid bins
    valid_mask = bins >= item
    
    if np.any(valid_mask):
        # For valid bins, score is the remaining space.
        # Higher score = more remaining space = Worst-Fit.
        remaining = bins[valid_mask] - item
        scores[valid_mask] = remaining
        
    return scores
