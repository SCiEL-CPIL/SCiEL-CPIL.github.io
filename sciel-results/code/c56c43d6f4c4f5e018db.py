"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 295.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Expected Improvement (EI) divided by a cost term.
    The cost term is raised to a power of 0.5 to reduce the bias towards
    cheaper (but potentially sub-optimal) regions, allowing the EI signal
    to dominate the search towards the true minimum.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple fallback: prefer low mu (exploitation) with cost awareness
        # Using sqrt(cost) in denominator for consistency with main logic
        return (f_best - mu) / (np.sqrt(cost) + 1e-9)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical overflow in norm.pdf/cdf
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: improvement per unit cost (with reduced cost penalty)
    # Using cost ** 0.5 instead of cost ** 1.0
    cost_term = np.sqrt(cost)
    
    cost_ei = ei / cost_term
    
    # Ensure all values are finite
    score = np.where(np.isfinite(cost_ei), cost_ei, 0.0)
    
    return score
# EVOLVE-BLOCK-END
