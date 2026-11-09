"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # 300s limit + 20s safety margin


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() > _DEADLINE:
        # Return a simple score based on mu (lower is better)
        return -mu
    
    # Expected improvement per unit cost with a small exploration bonus
    # Use a slightly modified EI that accounts for uncertainty
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost, but add a small exploration term
    # The exploration term encourages evaluating points with high uncertainty
    # when the expected improvement is low
    exploration = 0.1 * sigma / (cost + 1e-9)
    
    score = ei / cost + exploration
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
