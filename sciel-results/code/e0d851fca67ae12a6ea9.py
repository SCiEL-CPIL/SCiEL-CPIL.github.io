"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity for each bin
    rem = bins - item
    
    # Initialize scores with -inf to handle invalid bins (not enough capacity)
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Mask for bins that can fit the item
    valid = rem >= 0
    
    if np.any(valid):
        r = rem[valid]
        # Base score: Best-Fit (prefer smaller remainder)
        # We want to maximize score, so -rem is the base.
        base_score = -r
        
        # Penalty for creating very small "dead" gaps.
        # If the remaining space is too small to likely fit future items,
        # it is effectively wasted. We penalize this to encourage
        # remainders that are more likely to be useful.
        # Using an exponential decay to smoothly penalize small remainders.
        # The scale 10.0 suggests we penalize remainders < ~10-15 units.
        # The coefficient 20.0 makes the penalty significant for small r.
        penalty = 20.0 * np.exp(-r / 10.0)
        
        scores[valid] = base_score - penalty
        
    return scores
# EVOLVE-BLOCK-END
