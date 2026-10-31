# EVOLVE-BLOCK-START
import numpy as np
import math

# Weibull parameters for the generated sequences (shape k, scale lambda)
# Assumed k=2.0, lambda=20.0 based on seed program and typical bin packing scales.
K_WEIBULL = 2.0
LAMBDA_WEIBULL = 20.0

def _median_next_item_size() -> float:
    """
    Calculate the median of the Weibull distribution.
    Median = lambda * (ln(2))^(1/k)
    This is a more robust central tendency measure for skewed distributions than the mean.
    """
    return LAMBDA_WEIBULL * (math.log(2)) ** (1.0 / K_WEIBULL)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Target Median Remaining Space.
    Instead of simply minimizing remaining space (Best Fit), which can create 
    unusable small gaps, we aim to leave a remaining space close to the median 
    size of the next item. This balances the need to close bins with the need 
    to keep space available for typical future items.
    
    Score = -abs(remaining_space_after - median_next_size)
    This gives the highest score (0) to bins that will have exactly the median 
    amount of space left, and penalizes both too little (wasted gap) and too 
    much (inefficient use of capacity) remaining space.
    """
    target_remaining = _median_next_item_size()
    
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Find valid bins: bins that can fit the current item
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
    
    # Calculate remaining space AFTER placing the item
    remaining = bins[valid_mask] - item
    
    # Score: negative of absolute deviation from target
    # We prefer remaining to be close to target_remaining.
    scores[valid_mask] = -np.abs(remaining - target_remaining)
    
    return scores
# EVOLVE-BLOCK-END
