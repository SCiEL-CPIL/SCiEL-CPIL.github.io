"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: return random scores or just EI
        # To be safe, return a simple EI
        eps = 1e-9
        sigma_safe = sigma + eps
        z_ei = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Combine Expected Improvement and UCB for balanced exploration/exploitation.
    eps = 1e-9
    sigma_safe = sigma + eps
    
    # Expected Improvement
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # UCB (Upper Confidence Bound) for minimization
    # Score = -mu + beta * sigma (higher is better for exploration)
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Normalize to [0, 1] range to balance scales
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    ei_range = ei_max - ei_min
    ucb_range = ucb_max - ucb_min
    
    ei_norm = (ei - ei_min) / (ei_range + eps)
    ucb_norm = (ucb - ucb_min) / (ucb_range + eps)
    
    # Weighted combination
    # Slightly favor EI as we have more data
    w_ei = 0.8
    w_ucb = 0.2
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
