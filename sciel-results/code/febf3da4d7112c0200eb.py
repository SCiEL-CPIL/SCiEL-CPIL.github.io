"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred.
    
    Strategy: Smooth best-fit with relative-waste penalty.
    - Primary: minimize absolute leftover space (best-fit)
    - Secondary: penalize high relative waste (item is small fraction of remaining)
    - Uses smooth surrogate to avoid discontinuities
    """
    n = len(bins)
    scores = np.full(n, -1e18)
    
    # Calculate remaining capacity after placement
    remaining = bins - item
    
    # Only consider bins where remaining >= 0
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_remaining = remaining[valid_mask]
    
    # Relative waste: how much of the remaining capacity is "wasted" relative to item size
    # If item is small compared to remaining, relative waste is high
    # We want to prefer bins where item fills a large fraction of remaining capacity
    
    # Smooth relative waste metric: 
    #   rel_waste = remaining / (remaining + item)
    # This is in [0, 1), high when remaining >> item (bad), low when remaining << item (good)
    # It's smooth and well-behaved, no division by zero issues
    
    # Combined score: minimize absolute remainder + penalize relative waste
    # score = -remaining - beta * rel_waste
    # For best-fit (small remaining): -remaining dominates
    # For large remaining: relative waste penalty kicks in
    
    # Use a smooth form that blends both objectives
    # The relative waste term is already smooth
    
    beta = 0.5  # weight for relative waste penalty
    rel_waste = valid_remaining / (valid_remaining + item)
    
    # Also add a mild superlinear penalty on very large remainders to further reduce fragmentation
    # This helps when multiple bins have similar relative waste but very different absolute remainders
    delta = 1.0
    alpha = 0.005
    smooth_penalty = alpha * valid_remaining * valid_remaining / (valid_remaining + delta)
    
    scores[valid_indices] = -valid_remaining - beta * rel_waste - smooth_penalty
    
    return scores
# EVOLVE-BLOCK-END
