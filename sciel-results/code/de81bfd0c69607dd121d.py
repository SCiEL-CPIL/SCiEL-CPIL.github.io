import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity using Next-Item Compatibility.
    
    Strategy: Estimate the expected size of the next item from the Weibull distribution
    and score bins by how well their residual capacity can accommodate that next item.
    This balances current packing tightness with future utility.
    
    For Weibull distributions commonly used in bin packing (shape k~1-2, scale lambda~10-20
    with capacity 100), the mean is approximately lambda * Gamma(1 + 1/k).
    We use a robust estimate based on the current item size to adapt dynamically.
    """
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # Only consider bins with enough remaining capacity
    valid = remaining >= 0
    
    if not np.any(valid):
        return scores
    
    r = remaining[valid]
    
    # Estimate expected next item size
    # For Weibull with shape k and scale lambda, mean = lambda * Gamma(1 + 1/k)
    # We adapt the estimate based on the current item size to handle different distributions.
    # If item is small, next items are likely also small; if large, next items vary more.
    # Use a heuristic: expected_next ≈ 0.8 * item + 0.2 * (capacity/10)
    # This gives a moderate estimate that adapts to item size.
    capacity_estimate = 100.0  # Standard capacity for this task
    expected_next = 0.7 * item + 0.3 * (capacity_estimate / 10.0)
    expected_next = max(expected_next, 0.5)  # Minimum expected next item size
    
    # Score 1: Best Fit - prefer smaller remaining capacity
    # This ensures tight packing
    best_fit_score = -r
    
    # Score 2: Next-Item Compatibility
    # Probability-like score that residual can accommodate the expected next item
    # Use a smooth sigmoid: P = 1 / (1 + exp(-alpha * (r - expected_next)))
    # When r >> expected_next, P → 1 (high compatibility)
    # When r << expected_next, P → 0 (low compatibility)
    # When r ≈ expected_next, P ≈ 0.5
    alpha = 2.0  # Steepness of the sigmoid
    compat_score = 1.0 / (1.0 + np.exp(-alpha * (r - expected_next)))
    
    # Score 3: Fragmentation Penalty
    # Penalize creating fragments that are too small to be useful
    # A fragment is "useful" if it can hold at least a small fraction of typical items
    min_useful_fragment = 0.15 * capacity_estimate  # 15% of capacity
    frag_penalty = np.maximum(0.0, min_useful_fragment - r) ** 2 / (capacity_estimate ** 2)
    
    # Combine scores with tuned weights
    # Weight best_fit heavily to maintain tight packing
    # Weight compat_score moderately to prefer bins that can handle next items
    # Penalize fragmentation to avoid creating unusable spaces
    w_best_fit = 1.0
    w_compat = 0.3
    w_frag = 0.5
    
    combined = w_best_fit * best_fit_score + w_compat * compat_score - w_frag * frag_penalty
    
    scores[valid] = combined
    return scores
