import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    Strategy: Hybrid Best/Worst Fit based on item size relative to capacity.
    - Large items (> 50): Worst Fit (leave max space to avoid fragmentation for other large items)
    - Small items (<= 50): Best Fit with sliver penalty (consolidate small items, avoid useless slivers)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full_like(bins, -1e9, dtype=float)
        valid = remaining >= 0
        # Fallback: simple best fit
        scores[valid] = -remaining[valid]
        return scores

    remaining = bins - item
    valid = remaining >= 0
    scores = np.full_like(bins, -1e9, dtype=float)
    
    if not np.any(valid):
        return scores

    valid_remaining = remaining[valid]
    valid_indices = np.where(valid)[0]

    # Determine strategy based on item size
    # Capacity is 100. 
    # If item > 50, it's a "large" item. We want to keep bins open for other large items -> Worst Fit.
    # If item <= 50, it's a "small" item. We want to pack tightly -> Best Fit with sliver avoidance.
    
    if item > 50.0:
        # Worst Fit: maximize remaining capacity
        # Score is proportional to remaining capacity
        scores[valid_indices] = valid_remaining
    else:
        # Best Fit with Sliver Avoidance
        # Base score: -remaining (lower remaining is better)
        # Penalty for slivers: if remaining is very small, it's likely useless.
        # We want to avoid creating slivers < 5% of capacity (5 units).
        # Score = -remaining - penalty
        
        base_score = -valid_remaining
        
        # Sliver threshold: 5 units (5% of 100)
        sliver_threshold = 5.0
        
        # Use a quadratic penalty to create a smooth, stronger repulsion from slivers.
        # This acts as a smooth surrogate for the hard constraint "do not create slivers".
        # penalty = k * (threshold - remaining)^2
        # With k=1.0:
        # rem=1: 1*(4^2) = 16. Score = -1 - 16 = -17
        # rem=4: 1*(1^2) = 1. Score = -4 - 1 = -5
        # rem=5: 0. Score = -5
        # rem=6: 0. Score = -6
        # Preference: 6 > 5 = 4 > 1. 
        # Wait, 5 and 4 have same score? 
        # Let's adjust k to differentiate.
        # If k=2.0:
        # rem=1: 2*16 = 32. Score = -33
        # rem=4: 2*1 = 2. Score = -6
        # rem=5: 0. Score = -5
        # rem=6: 0. Score = -6
        # Preference: 5 > 6 = 4 > 1.
        # This prefers 5 over 6, which is good for density, but avoids 4 and 1.
        
        penalty = np.zeros_like(valid_remaining)
        tight_mask = valid_remaining < sliver_threshold
        if np.any(tight_mask):
            diff = sliver_threshold - valid_remaining[tight_mask]
            penalty[tight_mask] = 2.0 * (diff ** 2)
            
        final_scores = base_score - penalty
        scores[valid_indices] = final_scores

    return scores
