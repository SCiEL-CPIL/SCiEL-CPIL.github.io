"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin using a Worst-Fit Decreasing style heuristic.
    
    The primary score rewards bins with larger remaining capacity, which
    preserves high-capacity bins for future large items. A secondary term
    provides a mild preference for tighter fits to avoid excessive empty
    space, but the dominant effect is worst-fit.
    
    Score = rem_norm * (1.0 - 0.1 * (1.0 - rem_norm))
    
    Where:
    - rem_norm = remaining / capacity
    - The linear term (rem_norm) drives worst-fit behavior
    - The quadratic correction mildly penalizes very empty bins
    """
    start_time = time.time()
    
    # Compute remaining capacity if item is placed in each bin
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Create scores array, default to very low for invalid bins
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    rem_valid = remaining[valid_mask]
    
    # Capacity is 100 per the task description
    capacity = 100.0
    
    # Normalize remaining by capacity
    rem_norm = rem_valid / capacity
    
    # Worst-fit dominant scoring with mild tight-fit correction
    # Primary: reward larger remaining capacity (worst-fit)
    primary = rem_norm
    
    # Secondary: mild penalty for very empty bins (to avoid wasting capacity)
    # This is a smooth correction that slightly favors tighter fits
    # without overriding the worst-fit preference
    beta = 0.1
    secondary = -beta * (1.0 - rem_norm) ** 2
    
    # Combined score: worst-fit dominates, with slight preference for
    # not leaving too much empty space
    scores[valid_mask] = primary + secondary
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 140:  # Safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
