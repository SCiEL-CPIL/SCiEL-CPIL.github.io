"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins using Next-Item Awareness heuristic.
    
    Instead of pure best-fit, we estimate the expected size of the next item
    and prefer bins whose remaining space (after placing current item) best
    matches that prediction. This reduces wasted capacity by keeping bins
    with "useful" remaining space for future items.
    """
    rem = bins - item
    
    # Mask for valid bins (enough capacity)
    valid = rem >= 0
    
    # For invalid bins, assign very low score
    scores = np.full_like(rem, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
    
    valid_rem = rem[valid]
    
    # Estimate target remaining space: use median of valid remainders as a
    # robust proxy for "typical" future item size / useful gap size.
    # This avoids being skewed by outliers (very full or very empty bins).
    if len(valid_rem) > 0:
        target = np.median(valid_rem)
    else:
        target = 0.0
    
    # Score: negative squared distance from target.
    # This creates a smooth preference for bins whose remaining space
    # is close to the estimated useful size, penalizing both too-small
    # (wasted) and too-large (underutilized) remainders.
    scores[valid] = -(valid_rem - target) ** 2
    
    return scores
# EVOLVE-BLOCK-END
