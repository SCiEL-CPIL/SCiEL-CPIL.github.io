"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return a simple, fast result if we are close to the 330s limit
    # 300s provides a 30s safety margin.
    if time.time() - _start_time > 300:
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the acquisition, so we use the negative of the lower confidence bound.
    # LCB = mu - beta * sigma. Maximize -LCB = beta * sigma - mu.
    # beta controls exploration. 2.0 is a standard choice.
    beta = 2.0
    ucb = beta * sigma_safe - mu
    
    # Blend EI and UCB
    # Based on measured knowledge: 0.6 UCB, 0.4 EI performed well (1.037 vs 1.0)
    blended_score = 0.4 * ei + 0.6 * ucb
    
    # Cost-aware adjustment
    # Higher score means more likely to be evaluated.
    # We divide by cost to prefer cheaper evaluations.
    # Cost is in [1, 2], so this is a mild penalty.
    final_score = blended_score / (cost + 1e-9)
    
    return final_score
# EVOLVE-BLOCK-END
