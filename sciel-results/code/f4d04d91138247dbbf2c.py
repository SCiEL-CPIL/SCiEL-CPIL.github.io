"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (cEI) with numerical safeguards.
    mu, sigma, f_best, cost: 1-D arrays of shape (2000,).
    Returns 2000 finite scores; higher = evaluate next.
    """
    # Guard against zero/negative sigma
    sig = np.maximum(sigma, 1e-9)
    
    # Standard normal CDF/PDF for EI
    z = (f_best - mu) / sig
    # Clip z to avoid numerical overflow in exp
    z = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    ei = (f_best - mu) * norm.cdf(z) + sig * norm.pdf(z)
    
    # Cost-aware: divide by cost (higher cost = lower priority)
    # Add small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
