"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Quantile-based acquisition function (Q0.1).
    Returns the 10th percentile of the posterior predictive distribution
    at each candidate point, negated so that higher score = more promising.
    """
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # The 10th percentile of a normal distribution N(mu, sigma^2)
    # is mu + z * sigma, where z is the 0.1 quantile of the standard normal
    # z_0.1 ≈ -1.2815515655446004
    z_01 = -1.2815515655446004
    
    # Q0.1 = mu + z_01 * sigma
    q01 = mu + z_01 * sigma_safe
    
    # We want to minimize the function, so lower Q0.1 is better.
    # Return negated so that higher score means evaluate next.
    return -q01
# EVOLVE-BLOCK-END
