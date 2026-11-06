import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity if item is placed in each bin
    remaining = bins - item
    
    # Mask for valid bins (enough capacity)
    valid_mask = remaining >= 0
    
    # Initialize scores to -infinity (or a very low number) for invalid bins
    # We want the highest score to be picked.
    # For invalid bins, score should be -inf so they are never picked.
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores

    # Adaptive threshold: 
    # If item is small, use Worst-Fit (maximize remaining space left).
    # If item is large, use Best-Fit (minimize remaining space left, i.e., maximize score = -remaining).
    
    # Let's use a relative threshold. Capacity is 100.
    # Weibull items often have a mode and tail.
    # A common split is around the mean or median.
    # Without knowing exact dist, let's try 20% of capacity as "small".
    threshold = 20.0 
    
    if item < threshold:
        # Small item: Prefer bins that leave the MOST space (Worst-Fit)
        # Score = remaining capacity. Higher is better.
        scores[valid_mask] = remaining[valid_mask]
    else:
        # Large item: Prefer bins that leave the LEAST space (Best-Fit)
        # Score = -remaining capacity. Higher (closer to 0) is better.
        scores[valid_mask] = -remaining[valid_mask]
        
    return scores
