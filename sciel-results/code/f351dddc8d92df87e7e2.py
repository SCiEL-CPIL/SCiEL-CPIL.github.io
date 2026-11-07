"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 295.0  # Safety margin before 300s/330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = delta * cdf + sigma_safe * pdf
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware score: EI per unit cost, plus a small exploration bonus
    # The exploration bonus encourages visiting cheap, uncertain regions
    # to improve the GP model
    exploration = sigma_safe * 0.1 / cost_safe
    
    # Combine: primary signal is EI/cost, with mild exploration incentive
    score = ei / cost_safe + exploration
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
