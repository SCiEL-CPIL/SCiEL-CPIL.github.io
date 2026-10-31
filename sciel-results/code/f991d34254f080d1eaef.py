import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin for the given item.
    Higher score is better.
    Uses a hybrid strategy:
    - For large items (> 50% of capacity), use Worst Fit (maximize remaining space).
    - For small items (<= 50% of capacity), use Best Fit (minimize remaining space).
    """
    n_bins = len(bins)
    # Initialize all scores to a very low value (invalid for selection)
    scores = np.full(n_bins, -1e9, dtype=np.float64)
    
    # Find bins that can accommodate the item
    valid_mask = bins >= item
    
    # If no valid bins, return the low scores (selection will fail or pick arbitrary, 
    # but evaluator should handle this or input is valid)
    if not np.any(valid_mask):
        return scores
        
    # Extract remaining capacities of valid bins
    valid_remaining = bins[valid_mask]
    
    # Capacity is 100. Threshold is 50.
    threshold = 50.0
    
    if item > threshold:
        # Large item: Worst Fit
        # We want the bin with the MOST remaining space.
        # So score is proportional to remaining space.
        scores[valid_mask] = valid_remaining
    else:
        # Small item: Best Fit
        # We want the bin with the LEAST remaining space (but still >= item).
        # So score is inversely proportional to remaining space.
        # Negating makes smaller remaining -> larger score.
        scores[valid_mask] = -valid_remaining
        
    return scores
