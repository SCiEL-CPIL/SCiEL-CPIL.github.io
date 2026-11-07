"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    For minimization, EI is defined as:
    EI(x) = E[max(f_best - f(x), 0)]
    
    Given a Gaussian process posterior with mean mu and standard deviation sigma,
    and assuming f(x) ~ N(mu, sigma^2), the EI is:
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma, Phi is the standard normal CDF,
    and phi is the standard normal PDF.
    
    This function returns the EI score for each candidate. Higher scores
    indicate higher expected improvement, so the candidate with the
    highest score is selected for evaluation.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate the z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in the tails of the normal distribution
    z = np.clip(z, -50, 50)
    
    # Calculate Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure the output is finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
