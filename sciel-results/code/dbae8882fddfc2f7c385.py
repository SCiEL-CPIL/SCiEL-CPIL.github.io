"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture in 2 dimensions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function.
    
    Returns P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    This aggressively exploits the current best region, which is well-suited
    for the fixed RBF landscape where the minimum is well-defined.
    """
    # Numerical safety: clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    # This is in [0, 1], higher means more likely to beat f_best
    pi = norm.cdf(z)
    
    # Return finite scores; clip to [0, 1] for safety
    return np.clip(pi, 0.0, 1.0)
# EVOLVE-BLOCK-END
