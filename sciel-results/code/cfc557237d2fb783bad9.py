import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins using a polarization heuristic.
    Encourages bins to become either full (low remaining capacity) or empty (high remaining capacity).
    This avoids the 'half-full' state which is inefficient for packing.
    
    For large items, this behaves like Best-Fit.
    For small items, it dynamically chooses between Best-Fit and Worst-Fit based on which leads to a more extreme state.
    """
    C = 100.0
    mid = C / 2.0
    
    # Initialize scores to -inf so invalid bins are never chosen
    scores = np.full(bins.shape, -np.inf)
    
    # Identify bins that can fit the item
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
        
    # Calculate remaining capacity if the item is placed in the valid bins
    remaining = bins[valid_mask] - item
    
    # Polarization Score: Maximize the squared distance from the midpoint.
    # This rewards states where remaining capacity is close to 0 (full) or close to C (empty).
    scores[valid_mask] = (remaining - mid) ** 2
    
    return scores
