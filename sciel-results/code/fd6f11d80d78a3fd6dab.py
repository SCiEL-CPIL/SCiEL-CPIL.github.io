# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import gamma

# Weibull parameters for the generated sequences (shape k, scale lambda)
# The task mentions "five generated Weibull sequences" but doesn't specify exact params.
# Common default Weibull distributions in bin packing literature often use k=2, lambda=1
# or similar. However, since the item sizes must fit in bins of capacity 100,
# and typical online bin packing benchmarks use small items (e.g., uniform or exponential),
# let's assume a Weibull distribution scaled to have mean around 10-20 items per bin.
# A Weibull with k=2, lambda=20 has mean = 20 * Gamma(1+1/2) = 20 * 0.886 = 17.7.
# This seems reasonable.

# Let's use k=2.0, lambda=20.0 as a default assumption for the Weibull distribution.
# If the actual distribution is different, this heuristic might be suboptimal,
# but it's a reasonable guess for "Weibull sequences of 1000 items with capacity 100".

K_WEIBULL = 2.0
LAMBDA_WEIBULL = 20.0
CAPACITY = 100.0

def _expected_next_item_size() -> float:
    """
    Estimate the expected size of the next item.
    For a Weibull distribution, E[X] = lambda * Gamma(1 + 1/k).
    """
    return LAMBDA_WEIBULL * gamma(1 + 1.0 / K_WEIBULL)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Minimize the "effective waste".
    We define effective capacity as remaining_space - expected_next_item_size.
    We want to maximize the probability of fitting a second item.
    This is equivalent to minimizing the remaining space, but adjusted for the
    likelihood that a future item will fit.
    
    Actually, the directive says:
    "scoring each bin based on the negative of the 'effective capacity' defined as
    the remaining space minus the expected size of the next item... thereby
    prioritizing bins that maximize the probability of accepting a second consecutive item."
    
    So, score = -(remaining_space - expected_next_size)
    This means we prefer bins where remaining_space is close to expected_next_size.
    If remaining_space < expected_next_size, the score is positive (good).
    If remaining_space > expected_next_size, the score is negative (bad).
    
    However, we must only consider bins with enough remaining capacity for the CURRENT item.
    bins[bins < item] should be masked out.
    
    Let's refine:
    1. Identify valid bins: bins >= item.
    2. For valid bins, calculate remaining = bins - item.
    3. Calculate score = -(remaining - expected_next_size).
    4. For invalid bins, set score to -infinity.
    """
    expected_next = _expected_next_item_size()
    
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Find valid bins
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        # No valid bins; return all -inf (or any values, but -inf is safe)
        return scores
    
    # Calculate remaining space for valid bins
    remaining = bins[valid_mask] - item
    
    # Score: negative of (remaining - expected_next_size)
    # This encourages bins where remaining is close to expected_next_size.
    # If remaining is much larger, score is low (negative).
    # If remaining is close to expected_next_size, score is high (near 0 or positive).
    # If remaining is smaller than expected_next_size, score is positive.
    scores[valid_mask] = -(remaining - expected_next)
    
    return scores
# EVOLVE-BLOCK-END
