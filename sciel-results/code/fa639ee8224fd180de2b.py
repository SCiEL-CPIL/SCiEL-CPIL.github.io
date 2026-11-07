"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid of Probability of Improvement with exponential cost penalty
    and Expected Improvement per unit cost to balance exploration and exploitation
    under budget constraints.
    """
    # Deadline guard: track elapsed time
    t0 = time.time()
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) - in objective units
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # EI per unit cost (BOB approach)
    ei_per_cost = ei / cost_safe
    
    # PI with exponential cost penalty: sharper non-linear trade-off
    # The exponential penalty strongly favors cheap, high-probability improvements
    # Scale the exponent so that cost differences of ~1 matter
    cost_penalty = np.exp(-0.5 * cost_safe)
    pi_cost = pi * cost_penalty
    
    # Combine: primarily PI with exponential cost penalty, plus EI/cost
    # This creates a sharper, non-linear trade-off that strongly favors
    # cheap, high-probability improvements
    score = 0.7 * pi_cost + 0.3 * ei_per_cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin check
    if time.time() - t0 > 280:
        # Return simple cost-normalized EI as fallback
        fallback = ei / cost_safe
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback
    
    return score
# EVOLVE-BLOCK-END
