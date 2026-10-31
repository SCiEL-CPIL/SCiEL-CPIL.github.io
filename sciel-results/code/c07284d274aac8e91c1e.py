import numpy as np
import time

_START_TIME = time.time()
DEADLINE = 148.0
SAFETY_MARGIN = 2.0

# Weibull parameters for item size distribution
WEIBULL_K = 1.5
WEIBULL_LAMBDA = 20.0

# Capacity
C = 100.0

# Soft-min temperature for combining regret terms
T = 2.0

def _softmin(x: np.ndarray, t: float = T) -> np.ndarray:
    """Soft-min: -log-sum-exp(-x/t) * t"""
    x = np.asarray(x, dtype=float)
    m = np.min(x, axis=-1, keepdims=True)
    return -t * np.log(np.sum(np.exp(-(x - m) / t), axis=-1, keepdims=True))


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline check
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Fallback to simple Best Fit
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=float)
        if np.any(feasible):
            rem = bins[feasible] - item
            scores[feasible] = -rem
        return scores

    n_bins = len(bins)
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)

    scores = np.full_like(bins, -np.inf, dtype=float)

    rem = bins[feasible] - item

    # Weibull CDF: P(X <= rem) for X ~ Weibull(k, lambda)
    # This estimates the probability that a random future item fits in the remaining space
    weibull_cdf = 1.0 - np.exp(-np.power(rem / WEIBULL_LAMBDA, WEIBULL_K))

    # Regret 1: Wasted capacity – if rem is large, we may leave space unused
    # Use a smooth penalty that grows with rem
    # Normalize rem to [0, 1] by dividing by C
    rem_norm = rem / C

    # Regret 2: Fragmentation – if rem is too small to fit typical items, it's wasted
    # Penalty when rem is below the median of the Weibull distribution
    # Median of Weibull(k, lambda) = lambda * (ln 2)^(1/k)
    median_rem = WEIBULL_LAMBDA * (np.log(2.0)) ** (1.0 / WEIBULL_K)
    frag_penalty = np.maximum(0.0, median_rem - rem) / median_rem

    # Combine: we want to maximize weibull_cdf (fillability) while minimizing waste and fragmentation
    # Score = weibull_cdf - alpha * waste_penalty - beta * frag_penalty
    # waste_penalty: smooth function of rem_norm, higher when rem is large
    waste_penalty = rem_norm ** 1.5

    alpha = 0.3
    beta = 0.5

    combined = weibull_cdf - alpha * waste_penalty - beta * frag_penalty

    scores[feasible] = combined

    return scores
