"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Weibull parameters for item size distribution
# Shape k=2.0, scale lambda such that items fit in capacity 100
WEIBULL_K = 2.0
WEIBULL_LAMBDA = 12.0

# Mode of Weibull distribution: lambda * ((k-1)/k)^(1/k)
# For k=2: mode = lambda * (1/2)^(1/2) = lambda / sqrt(2)
WEIBULL_MODE = WEIBULL_LAMBDA * ((WEIBULL_K - 1.0) / WEIBULL_K) ** (1.0 / WEIBULL_K)

# Precompute constants for speed
_MODE = WEIBULL_MODE
_K = WEIBULL_K
_LAMBDA = WEIBULL_LAMBDA

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin by how well the resulting remainder aligns with the
    most probable future item size (mode of Weibull distribution).
    
    Strategy: Inventory model where we want to leave a gap that is either:
    1. Very small (bin nearly full) - low waste
    2. Close to the mode of the item distribution - high chance of perfect fit
    
    Score = -squared_deviation(remainder, mode) - alpha * remainder
    This penalizes remainders far from the mode, steering toward optimal gaps.
    """
    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Compute remaining capacity after placing the item
    remaining = bins - item
    
    # Clip to avoid negative values (shouldn't happen for valid bins)
    rem_clipped = np.maximum(remaining, 0.0)
    
    # Core idea: Steer remainders toward the mode of the Weibull distribution.
    # If remainder ≈ mode, the next item is very likely to fit perfectly.
    # If remainder << mode, the gap is too small (wasted space).
    # If remainder >> mode, we're leaving too much room (fragmentation).
    
    # Squared deviation from the mode
    dev_from_mode = (rem_clipped - _MODE) ** 2
    
    # Base penalty: we still want to minimize remaining space overall
    # But the squared deviation term dominates the behavior
    
    # Weight for the deviation term vs the linear waste term
    # Higher weight on deviation means stronger steering toward the mode
    alpha = 0.1  # linear waste penalty weight
    
    # Score: negative of (deviation penalty + linear waste penalty)
    # We want to MINIMIZE this, so we return negative scores
    scores = -(dev_from_mode + alpha * rem_clipped)
    
    # Mask invalid bins
    scores[~valid] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
