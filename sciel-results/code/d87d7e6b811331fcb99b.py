"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Standard EI per unit cost
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_cost = ei / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(ei_cost), ei_cost, 0.0)
    
    # Check deadline - if we're getting close to timeout, return immediately
    elapsed = time.time() - start_time
    if elapsed > 290:
        return score
    
    return score
# EVOLVE-BLOCK-END
