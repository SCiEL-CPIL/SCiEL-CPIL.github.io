"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Weibull distribution parameters for the generated sequences
# Based on typical online bin packing benchmarks with capacity 100
# We estimate the mean item size to be around 10-20 for Weibull distributions
# used in such benchmarks. Let's use a reasonable default.
# The mean of a Weibull distribution with shape k and scale lambda is lambda * Gamma(1 + 1/k)
# For k=1 (exponential), mean = lambda
# For k=2 (Rayleigh), mean = lambda * sqrt(pi)/2 ≈ lambda * 0.886
# Common Weibull parameters in bin packing: shape ~1.5-2.5, scale chosen so mean is reasonable

# We'll use adaptive estimation based on observed items
# Start with a reasonable default mean
_DEFAULT_MEAN = 15.0  # Estimated mean item size

# Track state for adaptive mean estimation
_state = {
    'sum': 0.0,
    'count': 0,
    'start_time': None
}


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _state
    
    # Deadline guard: if we've been running too long, fall back to simple heuristic
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    
    elapsed = time.time() - _state['start_time']
    
    # Update adaptive mean estimate
    _state['sum'] += item
    _state['count'] += 1
    current_mean = _state['sum'] / _state['count'] if _state['count'] > 0 else _DEFAULT_MEAN
    
    # Use a smoothed estimate (exponential moving average)
    # Blend with default to avoid extreme values early on
    alpha = min(1.0, _state['count'] / 50.0)  # More weight on observed data as count grows
    mean_est = alpha * current_mean + (1 - alpha) * _DEFAULT_MEAN
    
    # Ensure mean_est is reasonable
    mean_est = max(1.0, min(mean_est, 50.0))
    
    # Simple fallback if time is running out
    if elapsed > 100:  # Safety margin before 120s limit
        return -(bins - item)
    
    n_bins = len(bins)
    scores = np.zeros(n_bins, dtype=np.float64)
    
    # Calculate remaining capacity for each bin
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        # No valid bins - this shouldn't happen if the evaluator handles it
        return scores
    
    # For each valid bin, compute a score based on "next-item compatibility"
    # The idea: after placing the current item, the remaining capacity should be
    # likely to accommodate future items. We estimate the probability that the
    # remaining capacity is at least the mean expected next item size.
    
    # Remaining capacity after placing current item
    rem_after = remaining[valid_mask]
    
    # Score components:
    # 1. Tightness: prefer bins that leave less waste (lower remaining is better)
    # 2. Next-item compatibility: prefer bins where remaining capacity >= mean_est
    #    with some probability margin
    
    # Component 1: Negative remaining capacity (tighter fit is better)
    tightness_score = -rem_after
    
    # Component 2: Probability-based score
    # If remaining >= mean_est, high score; if remaining < mean_est, lower score
    # Use a smooth transition
    # P(remaining >= mean_est) approximated by a sigmoid-like function
    # Scale: if rem_after is well above mean_est, score ~1; if well below, score ~0
    # Use a logistic function centered at mean_est
    scale = max(mean_est * 0.5, 1.0)  # Scale for the transition width
    compat_score = 1.0 / (1.0 + np.exp(-(rem_after - mean_est) / scale))
    
    # Component 3: Dead zone penalty
    # Penalize remainders that are in the "dead zone": too small to fit a typical
    # future item but not small enough to be negligible
    # Dead zone: rem_after < mean_est * 0.5 but rem_after > 0
    # This is where capacity is wasted
    dead_zone_start = 0.0
    dead_zone_end = mean_est * 0.3
    # Penalty: high when rem_after is in the dead zone
    # Use a smooth bump function
    if dead_zone_end > dead_zone_start:
        # Gaussian-like bump centered in the dead zone
        center = (dead_zone_start + dead_zone_end) / 2
        width = (dead_zone_end - dead_zone_start) / 2
        if width > 0:
            dead_zone_penalty = np.exp(-((rem_after - center) ** 2) / (2 * width ** 2))
        else:
            dead_zone_penalty = np.zeros_like(rem_after)
    else:
        dead_zone_penalty = np.zeros_like(rem_after)
    
    # Combine scores
    # Weighting: tightness (0.3), compatibility (0.4), dead zone penalty (-0.3)
    combined = 0.3 * tightness_score + 0.4 * compat_score - 0.3 * dead_zone_penalty
    
    # Normalize tightness to be comparable in scale
    max_tightness = np.max(np.abs(tightness_score)) if len(tightness_score) > 0 else 1.0
    if max_tightness > 0:
        combined = 0.3 * (tightness_score / max_tightness) + 0.4 * compat_score - 0.3 * dead_zone_penalty
    
    # Assign scores to valid bins
    scores[valid_mask] = combined
    
    return scores
# EVOLVE-BLOCK-END
