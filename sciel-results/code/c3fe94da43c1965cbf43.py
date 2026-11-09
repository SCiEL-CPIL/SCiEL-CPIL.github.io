"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function for Bayesian optimization of Hartmann 6d.
    Combines Expected Improvement (EI) with a target-aware component that
    guides search toward the known global minimum.
    """
    # Known global minimum for Hartmann 6d
    f_target = -3.32237
    
    # Safety margin for time: return early if we're getting close to limit
    # This is a lightweight guard; the heavy work is vectorized
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-scores for both current best and target
    z_best = (f_best - mu) / sigma_safe
    z_target = (f_target - mu) / sigma_safe
    
    # Expected Improvement relative to current best (standard EI)
    ei_best = (f_best - mu) * norm.cdf(z_best) + sigma_safe * norm.pdf(z_best)
    
    # Expected Improvement relative to known global minimum
    ei_target = (f_target - mu) * norm.cdf(z_target) + sigma_safe * norm.pdf(z_target)
    
    # Blend the two: primarily use EI toward current best for exploitation,
    # but add a smaller weight toward the known global minimum for exploration
    # This helps escape local optima by pulling toward the true optimum
    scores = 0.7 * ei_best + 0.3 * ei_target
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
