"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    Parameters:
        mu: array of shape (2000,) with posterior mean values
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: scalar, the best (lowest) observed function value so far
    
    Returns:
        scores: array of shape (2000,) with acquisition scores (higher = better)
    """
    # Ensure sigma is strictly positive to avoid division by zero
    # In BO, sigma should be > 0, but numerical issues can occur
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the Z-score: (f_best - mu) / sigma
    # If mu < f_best, Z is positive (potential improvement)
    # If mu > f_best, Z is negative (unlikely to improve)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    scores = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure all scores are finite
    # If any NaN or Inf crept in, replace with 0 (neutral) or a very small negative number
    # EI should theoretically be non-negative, but numerical errors might cause tiny negatives
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
