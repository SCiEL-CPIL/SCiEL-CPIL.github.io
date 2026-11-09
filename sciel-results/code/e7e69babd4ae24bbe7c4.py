"""Rewritten program using Expected Improvement (EI) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred for evaluation.
    """
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical instability
    # A small epsilon prevents log(0) or inf values
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Z-score for Expected Improvement
    # Z = (f_best - mu) / sigma
    # We use f_best as the reference for improvement
    z = (f_best - mu) / sigma_safe
    
    # To handle large positive Z (where mu << f_best), we can just return the gain, 
    # but the standard formula works well. 
    # To handle large negative Z (where mu >> f_best), EI should be ~0.
    
    # Compute the standard normal CDF and PDF
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    
    # Handle potential numerical issues with very large z
    # If z is very large positive, cdf ~ 1, pdf ~ 0. EI ~ f_best - mu.
    # If z is very large negative, cdf ~ 0, pdf ~ 0. EI ~ 0.
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure all scores are finite
    # Sometimes numerical errors can produce nan or inf
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # EI is always non-negative. 
    # We return scores as is, since higher EI is better.
    
    return scores
# EVOLVE-BLOCK-END
