import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity
    remaining = bins - item
    
    # Mask invalid bins (just in case, though likely pre-filtered)
    valid_mask = remaining >= 0
    
    # Initialize scores to -inf to ensure invalid bins are not chosen
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores

    valid_remaining = remaining[valid_mask]
    
    # Hybrid Strategy:
    # If the item is large (> 50% of capacity), use Worst-Fit (maximize remaining space)
    # to keep large spaces available for other large items.
    # If the item is small, use Best-Fit (minimize remaining space) to fill gaps.
    
    if item > 50:
        # Worst-Fit: Higher score for larger remaining space
        scores[valid_mask] = valid_remaining
    else:
        # Best-Fit: Higher score for smaller remaining space
        scores[valid_mask] = -valid_remaining
        
    return scores
