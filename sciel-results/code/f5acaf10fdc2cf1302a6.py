"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses Probability of Improvement (PI) for minimization.
    PI favors points with a high probability of being better than f_best.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement (PI) for minimization
    # PI = Phi((f_best - mu) / sigma)
    # where z = (f_best - mu) / sigma, Phi is standard normal CDF
    z = (f_best - mu) / sigma_safe
    
    # Calculate PI
    pi = norm.cdf(z)
    
    # Ensure PI is within [0, 1] for numerical stability
    pi = np.clip(pi, 0.0, 1.0)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(pi), pi, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
