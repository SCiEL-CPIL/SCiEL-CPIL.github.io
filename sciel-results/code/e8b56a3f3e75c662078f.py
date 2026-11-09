"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin using a nonlinear Best Fit heuristic.
    
    This function uses a power-law transformation of the remaining capacity to
    score bins. The score is higher (less negative) for bins that leave less
    remaining space after placing the item. The exponent alpha controls the
    sensitivity to tight fits. A value of alpha < 1.0 makes the heuristic more
    aggressive in selecting the tightest possible fit, which can reduce
    fragmentation for skewed item size distributions like Weibull.
    """
    start_time = time.time()
    
    # Capacity is fixed at 100 per task description
    capacity = 100.0
    
    # Compute remaining capacity if item is placed in each bin
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Create scores array, default to very low for invalid bins
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    # Normalize remaining capacity
    # Ensure we don't divide by zero or have negative values in valid mask
    u = remaining[valid_mask] / capacity
    
    # Nonlinear power-law scoring
    # alpha = 0.7 is chosen based on inspiration 2 and cell findings suggesting
    # that a smooth nonlinear transform with alpha < 1.0 improves performance
    # over the linear Best Fit (alpha = 1.0) for Weibull sequences.
    alpha = 0.7
    
    # Score: -u^alpha
    # Smaller u (tighter fit) -> higher score (closer to 0)
    # Larger u (looser fit) -> lower score (closer to -1)
    scores[valid_mask] = -(u ** alpha)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 140:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
