"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0  # seconds
_MARGIN = 5.0      # safety margin before deadline


def acquisition(mu, sigma, f_best, cost):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _MARGIN:
        # Fallback: return a simple cost-aware PI to be safe
        z = (f_best - mu) / (sigma + 1e-9)
        return norm.cdf(z) / (cost + 1e-9)

    # Clamp sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Risk-Averse UCB:
    # Exploitation: f_best - mu (higher is better, i.e., mu is lower)
    # Exploration: beta * sigma (higher is better, i.e., more uncertain)
    # Risk Penalty: - gamma * sigma^2 (penalizes very high uncertainty)
    
    # Tuned parameters for 8D Cosine8
    beta = 1.5
    gamma = 2.0
    
    # Calculate the core score
    # We want to maximize this score.
    # If mu is close to f_best, (f_best - mu) is small.
    # If sigma is moderate, beta * sigma adds value.
    # If sigma is large, - gamma * sigma^2 dominates and reduces score.
    
    score = (f_best - mu) + beta * sigma - gamma * sigma * sigma
    
    # Cost normalization: divide by cost to favor cheaper evaluations
    # cost is in [1, 2]
    score = score / (cost + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
