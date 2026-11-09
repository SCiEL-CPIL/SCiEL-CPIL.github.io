import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Uses a refined hybrid heuristic:
    1. Primary: Best Fit (minimize remaining capacity).
    2. Secondary: Smooth "Just Right" Gap Preference (Gaussian bump centered at 0.1*item).
    3. Tertiary: Large Gap Penalty (quadratic penalty for gaps > 1.2*item).
    
    The goal is to balance tight packing with avoiding fragmentation by
    preferring gaps that are small enough not to waste space but large enough
    to potentially accommodate future small items (if such items arrive),
    while strongly penalizing large gaps that waste significant capacity.
    """
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # Only consider bins with enough remaining capacity
    valid = remaining >= 0
    
    if not np.any(valid):
        return scores
    
    r = remaining[valid]
    
    # Primary: Best Fit (smaller remaining is better)
    # Negate so that higher score = better fit
    base_score = -r
    
    # Secondary: Gaussian bump for "slightly loose" fits
    # Center at 0.1 * item to encourage a very small but non-trivial gap.
    # Sigma is 0.05 * item to cover a narrow range of "good" gaps.
    center = 0.1 * item
    sigma = 0.05 * item + 1e-10
    bump = np.exp(-((r - center) ** 2) / (2.0 * sigma ** 2))
    
    # Tertiary: Large Gap Penalty
    # Penalize bins where the remaining capacity is significantly larger than the item.
    # This prevents placing small items in large bins, which wastes space.
    # We use a quadratic penalty that starts at 1.2 * item.
    # This is smoother than a linear penalty and grows faster for very large gaps.
    gap_threshold = 1.2 * item
    excess_gap = np.maximum(r - gap_threshold, 0.0)
    # Normalize the penalty by item size to make it scale-invariant
    # and add a coefficient to make it significant.
    # The base_score is linear in r. A quadratic penalty in (r/item) 
    # will eventually dominate for large r.
    penalty = (excess_gap / (item + 1e-10)) ** 2 * 0.5
    
    # Combine scores
    # The bump is in [0, 1]. We want it to have a moderate influence.
    # If weight is too high, we might prefer a gap of 0.1*item over 0.0*item.
    # base_score diff: -0.1*item - 0 = -0.1*item.
    # bump diff: 1.0 - ~0 = 1.0.
    # If weight > 0.1*item, we prefer the looser fit.
    # For item=1.0, weight > 0.1. For item=0.1, weight > 0.01.
    # A weight of 0.2 seems reasonable to encourage the "just right" gap
    # when the difference in remaining capacity is small relative to the item size.
    weight = 0.2
    
    combined = base_score + weight * bump - penalty
    
    scores[valid] = combined
    return scores
