import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the arriving item.
    The item is placed in the highest-scoring bin.
    
    We use a Best-Fit heuristic: prefer the bin where the item fits with
    the least wasted space (i.e., smallest remaining capacity after placement).
    This is known to perform well for online bin packing with random item sizes.
    
    For bins that cannot fit the item, we assign -inf so they are never chosen.
    For valid bins, score = -(bins - item), so higher scores correspond to
    smaller remaining space (tighter fit).
    """
    valid = bins >= item
    scores = np.full_like(bins, -np.inf, dtype=float)
    if np.any(valid):
        scores[valid] = item - bins[valid]
    return scores
