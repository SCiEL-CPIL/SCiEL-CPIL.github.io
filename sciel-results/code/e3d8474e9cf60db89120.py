"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    EI provides a principled way to balance exploration and exploitation
    by maximizing the expected reduction in the best observed value.
    
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: lowest observed value so far
    
    Returns: (2000,) array of acquisition scores (higher = more promising)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement: f_best - mu
    # We are minimizing, so improvement is positive if mu < f_best
    improvement = f_best - mu
    
    # Standard normal CDF and PDF
    # z = improvement / sigma
    z = improvement / sigma
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal distribution
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    score = improvement * Phi + sigma * phi
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
