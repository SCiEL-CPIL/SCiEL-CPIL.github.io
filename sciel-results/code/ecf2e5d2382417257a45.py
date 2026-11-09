"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Deadline guard
    deadline = start_time + 290

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Structural Change: Rank/Quantile-based normalization of PI
    # This decouples the absolute magnitude of improvement probability from the score,
    # focusing on the relative ranking of candidates.
    pi_min = np.min(pi)
    pi_max = np.max(pi)
    
    # Avoid division by zero
    pi_range = pi_max - pi_min
    if pi_range < 1e-9:
        pi_norm = np.ones_like(pi)
    else:
        pi_norm = (pi - pi_min) / pi_range
    
    # Cost-aware scoring: Linear cost penalty
    # The directive suggests "ratio of posterior probability ... to cumulative cost"
    score = pi_norm / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return pi_norm
    
    return score
# EVOLVE-BLOCK-END
