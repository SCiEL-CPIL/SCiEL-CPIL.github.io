"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    remaining = bins - item
    valid = remaining >= 0
    
    # Adaptive strategy: Worst Fit for large items, Best Fit for small items
    # Large items (> capacity/3 ~ 33.33): prefer bins with more remaining space
    # Small items: prefer bins with less remaining space (best fit)
    
    if item > 33.33:
        # Worst Fit: maximize remaining capacity
        # Higher remaining = better score
        score = remaining.copy()
        
        # Add a small penalty for very full bins to avoid fragmentation
        # Use soft-min surrogate: smooth approximation of min(remaining, threshold)
        threshold = 50.0  # Prefer bins with at least 50 remaining if possible
        # Soft-max: smooth approximation that prefers larger remaining
        # Using log-sum-exp style smoothing
        alpha = 0.1
        # Score increases with remaining, but with diminishing returns
        score = np.log1p(np.maximum(remaining, 0) * alpha)
        
    else:
        # Best Fit: minimize remaining capacity
        # Higher score = smaller remaining
        score = -remaining.copy()
        
        # Penalize very small remainders to avoid creating unusable gaps
        threshold = max(2.0, 0.05 * item)
        
        # Smooth penalty using soft-min surrogate
        # For remaining < threshold, apply penalty
        diff = threshold - remaining
        # Smooth quadratic penalty
        penalty = np.where(diff > 0, diff * diff * 0.1, 0.0)
        
        # Add penalty to score (reduces score for small remainders)
        score = score - penalty
    
    # Invalidate bins that can't fit the item
    score = np.where(valid, score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
