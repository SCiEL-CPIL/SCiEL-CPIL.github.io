import numpy as np
import time

# We assume capacity is 100 as per task description
CAPACITY = 100.0
START_TIME = time.time()
DEADLINE = 110.0  # Safety margin

# Global state to track the distribution of bin remaining spaces
# This is used for the "Next-Item Awareness" heuristic
_remaining_history = []
_mean_remaining = CAPACITY / 2.0  # Initial estimate
_std_remaining = CAPACITY / 3.0    # Initial spread

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _remaining_history, _mean_remaining, _std_remaining
    
    # Check deadline
    if time.time() - START_TIME > DEADLINE:
        # Fallback to simple best-fit if we're running out of time
        remaining = bins - item
        scores = np.full_like(bins, -np.inf, dtype=float)
        valid_mask = remaining >= 0
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return scores
        scores[valid_idx] = -remaining[valid_idx]
        return scores

    # Calculate remaining space after placing item
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Find valid indices (bins with enough remaining capacity)
    valid_mask = remaining >= 0
    valid_idx = np.where(valid_mask)[0]
    
    if len(valid_idx) == 0:
        return scores

    valid_remaining = remaining[valid_idx]
    
    # Update global statistics of remaining spaces
    # We track the current state of all bins' remaining space (before placing item)
    current_remaining = bins
    _remaining_history.append(current_remaining.copy())
    
    # Keep only recent history to avoid memory bloat
    if len(_remaining_history) > 100:
        _remaining_history = _remaining_history[-100:]
    
    # Compute mean and std of all remaining spaces in current bins
    # This gives a sense of the "typical" remaining space
    all_rem = bins
    _mean_remaining = np.mean(all_rem)
    _std_remaining = np.std(all_rem)
    if _std_remaining < 1e-6:
        _std_remaining = 1.0
    
    # Strategy: Next-Item Awareness
    # Estimate the expected size of the next item from the Weibull distribution
    # For Weibull with shape k and scale lambda, mean = lambda * Gamma(1 + 1/k)
    # We don't know the exact parameters, but we can estimate from the data
    # Alternatively, use the current item size as a proxy and blend with bin stats
    
    # The idea: prefer bins where the remaining space after placing the current item
    # is close to the expected next item size, to minimize waste
    # Expected next item size: use a smoothed estimate
    # For Weibull, the distribution is skewed, so median < mean
    # Use a conservative estimate: median of recent items or mean
    
    # Since we don't have access to the item sequence directly in a stateful way,
    # we use the current item and the bin occupancy distribution to estimate
    # A simple proxy: the mean remaining space of valid bins, or the current item size
    expected_next = _mean_remaining * 0.5  # Heuristic: next item likely smaller than avg remaining
    # Blend with current item size
    expected_next = 0.5 * item + 0.5 * expected_next
    
    # Score each valid bin: how well does its remaining space match expected_next?
    # We want remaining close to expected_next (not too much, not too little)
    # Use a Gaussian-like score: exp(-((remaining - expected_next)^2) / (2 * std^2))
    # Lower remaining difference -> higher score
    
    diff = valid_remaining - expected_next
    score_density = np.exp(-(diff ** 2) / (2 * (_std_remaining ** 2)))
    
    # Combine with best-fit: prefer smaller remaining (tighter packing)
    # Normalize valid_remaining to [0, 1]
    min_rem = np.min(valid_remaining)
    max_rem = np.max(valid_remaining)
    range_rem = max_rem - min_rem if max_rem > min_rem else 1.0
    normalized_rem = (valid_remaining - min_rem) / range_rem
    
    # Blend: 60% best-fit, 40% next-item awareness
    scores[valid_idx] = -0.6 * normalized_rem + 0.4 * score_density
    
    return scores
