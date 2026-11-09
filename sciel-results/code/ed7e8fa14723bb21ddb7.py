import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity using a stochastic
    inventory model based on the Weibull distribution.
    
    The score reflects the expected future waste reduction of preserving
    the remaining gap, weighted by the probability that the gap will be
    filled by future items (Weibull CDF).
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    remaining = bins - item
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    valid = remaining >= 0
    
    if not np.any(valid):
        return scores
    
    r = remaining[valid]
    
    if time.time() > deadline:
        scores[valid] = -r
        return scores
    
    positive_bins = bins[bins > 0]
    if len(positive_bins) < 2:
        scores[valid] = -r
        return scores
    
    median = np.median(positive_bins)
    k_est = 2.0
    lambda_est = median / (np.log(2) ** (1.0 / k_est))
    lambda_est = max(lambda_est, 1e-10)
    
    x = r / lambda_est
    cdf = 1.0 - np.exp(-x ** k_est)
    
    scores[valid] = -r * (1.0 - cdf)
    
    return scores
