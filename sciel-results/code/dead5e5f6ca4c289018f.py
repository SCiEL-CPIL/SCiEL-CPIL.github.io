import numpy as np
import time

# Global start time for the deadline guard
_start_time = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0  # Seconds

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity. Higher score is better.
    The item goes to the highest-scoring bin.
    
    Uses Best Fit strategy: minimize the remaining capacity after placing the item.
    This is robust for Weibull distributions.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
        # Return a default "safe" score, e.g., Best Fit or First Fit
        # To be safe, just return -remaining for valid, -inf for invalid
        remaining = bins - item
        valid_mask = remaining >= 0
        scores = np.full_like(remaining, -np.inf, dtype=float)
        if np.any(valid_mask):
            scores[valid_mask] = -remaining[valid_mask]
        return scores

    # Track remaining capacity for each bin
    remaining = bins - item
    
    # Filter to only bins that can hold the item
    valid_mask = remaining >= 0
    
    # Create score array with -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    n_valid = len(valid_remaining)
    
    if n_valid == 0:
        return scores
    
    if n_valid == 1:
        scores[valid_mask] = 0.0
        return scores
    
    # Best Fit: Minimize remaining capacity.
    # Score = -remaining.
    # This favors bins where the item fits tightly.
    
    # To break ties deterministically and prefer lower indices (First Fit among ties),
    # we can add a very small negative value proportional to index.
    # However, np.argmax picks the first max, so if scores are identical, index 0 is picked.
    # So simple -remaining is sufficient.
    
    scores[valid_mask] = -valid_remaining
    
    return scores
