"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PoI)
    per unit cost. This is a distinct approach from Expected Improvement (EI),
    focusing on the likelihood of finding a point better than the current best
    rather than the expected magnitude of improvement.
    """
    sigma_safe = np.maximum(sigma, 1e-6)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Probability of Improvement
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Cost-aware: PoI per unit cost
    score = poi / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
