"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import weibull_min

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()

# Pre-compute Weibull parameters based on the task description (5 sequences, capacity 100)
# The exact distribution isn't given, but Weibull with shape ~1.5-2.0 and scale ~30-40 
# is typical for such packing problems. We'll use a reasonable default.
# Since we don't know the exact params, we'll estimate or use a standard guess.
# For Weibull sequences in bin packing with capacity 100, a common setup is:
# shape (k) ~ 1.5, scale (lambda) ~ 30, so mean ~ 30 * Gamma(1+1/1.5) ~ 30 * 0.9 ~ 27
# Let's use k=1.5, lam=30 as a reasonable prior.
_WEIBULL_K = 1.5
_WEIBULL_LAM = 30.0

def _expected_waste(remaining: float) -> float:
    """
    Estimate the expected wasted space in a bin with 'remaining' capacity.
    This is the expected value of 'remaining' * P(no future item fits in 'remaining').
    Since items arrive from a Weibull distribution, the probability that a random
    item does NOT fit in 'remaining' space is P(X > remaining) = 1 - CDF(remaining).
    The expected waste is roughly remaining * (1 - CDF(remaining)).
    """
    if remaining <= 0:
        return 0.0
    # Probability that a single future item does not fit
    p_not_fit = 1.0 - weibull_min.cdf(remaining, _WEIBULL_K, scale=_WEIBULL_LAM)
    # Expected waste: if it doesn't fit, the entire 'remaining' is wasted.
    # This is a lower bound on waste; a more accurate model would consider
    # that multiple items might eventually fit, but for a greedy step,
    # this captures the immediate fragmentation risk.
    return remaining * p_not_fit


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        # Fallback: Best Fit
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    valid = bins >= item
    if not np.any(valid):
        return np.full_like(bins, -np.inf)

    n_bins = len(bins)
    scores = np.full(n_bins, -np.inf)

    # For each valid bin, compute the remaining space after placing the item
    remaining = bins[valid] - item

    # Score is the negative of the expected waste.
    # We want to minimize waste, so we assign higher scores to bins with lower expected waste.
    # waste_i = remaining_i * P(X > remaining_i)
    # score_i = -waste_i
    #
    # To make it more effective, we can also add a small tie-breaker favoring
    # larger remaining space (Worst Fit) if expected wastes are very close,
    # but let's stick to the pure regret model first.
    #
    # Actually, a better score is to minimize the expected waste.
    # So score = -expected_waste(remaining)
    
    # Vectorized computation
    # CDF for each remaining space
    cdf_vals = weibull_min.cdf(remaining, _WEIBULL_K, scale=_WEIBULL_LAM)
    p_not_fit = 1.0 - cdf_vals
    expected_waste = remaining * p_not_fit

    # We want to pick the bin with the MINIMUM expected waste.
    # Since the evaluator picks the HIGHEST score, we negate the waste.
    scores[valid] = -expected_waste

    return scores
# EVOLVE-BLOCK-END
