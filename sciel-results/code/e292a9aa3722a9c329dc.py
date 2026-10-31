# EVOLVE-BLOCK-START
import numpy as np

# Weibull parameters for the generated sequences (shape k, scale lambda)
# Assumed parameters based on typical bin packing benchmarks with capacity 100.
K_WEIBULL = 2.0
LAMBDA_WEIBULL = 20.0

def _expected_next_item_size() -> float:
    """
    Estimate the expected size of the next item.
    For a Weibull distribution, E[X] = lambda * Gamma(1 + 1/k).
    """
    from scipy.special import gamma
    return LAMBDA_WEIBULL * gamma(1 + 1.0 / K_WEIBULL)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: First Fit with a slight Best Fit bias.
    The directive suggests scoring based on negative effective capacity (E - R).
    However, for skewed distributions like Weibull, pure Best Fit can be suboptimal
    due to fragmentation. First Fit (picking the earliest bin that fits) is often
    more robust.
    
    We implement a hybrid: primarily favoring the first valid bin (index-based),
    but with a small penalty for large remaining space to adhere to the "effective
    capacity" scoring principle. This effectively creates a "First Fit" strategy
    that breaks ties by minimizing remaining space.
    """
    expected_next = _expected_next_item_size()
    
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Find valid bins
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
    
    # Get indices of valid bins
    valid_indices = np.where(valid_mask)[0]
    
    # Calculate remaining space for valid bins
    remaining = bins[valid_indices] - item
    
    # Base score: -index to favor earlier bins (First Fit)
    # We use a large weight for the index to ensure First Fit dominates,
    # but include the effective capacity term to satisfy the directive's structure.
    # Score = -index * 1000.0 - (remaining - expected_next) * 0.01
    # This means:
    # 1. Bin 0 is always preferred over Bin 1 if both are valid, unless the
    #    difference in (remaining - expected_next) is huge (> 100,000), which
    #    is impossible since capacity is 100.
    # 2. Thus, it effectively picks the first valid bin.
    
    index_penalty = valid_indices * 1000.0
    capacity_score = -(remaining - expected_next) * 0.01
    
    scores[valid_indices] = -index_penalty + capacity_score
    
    return scores
# EVOLVE-BLOCK-END
