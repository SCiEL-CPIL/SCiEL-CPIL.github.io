# EVOLVE-BLOCK-START
import numpy as np
import scipy.special

# Weibull parameters
K_WEIBULL = 2.0
LAMBDA_WEIBULL = 20.0

# Pre-calculate the "target" remaining space.
# The heuristic aims to leave a remaining space close to the "expected" next item size.
# Using the Mode (most frequent size) instead of the Mean can lead to tighter packing
# as it targets the most likely item size, reducing the chance of large gaps.
# Mode of Weibull(k, L) = L * (1 - 1/k)^(1/k)
_TARGET_SIZE = LAMBDA_WEIBULL * (1.0 - 1.0 / K_WEIBULL) ** (1.0 / K_WEIBULL)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Minimize the deviation of the remaining space from the target size.
    Score = -(remaining - target_size).
    This prioritizes bins where the remaining space after placing the item
    is close to the most likely size of the next item (Mode).
    """
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Find valid bins
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
    
    # Calculate remaining space for valid bins
    remaining = bins[valid_mask] - item
    
    # Score: negative of (remaining - target_size)
    # Maximizing this score minimizes |remaining - target_size| 
    # (specifically, it minimizes remaining if remaining < target, 
    # and penalizes remaining > target).
    # This is a directed Best Fit.
    scores[valid_mask] = -(remaining - _TARGET_SIZE)
    
    return scores
# EVOLVE-BLOCK-END
