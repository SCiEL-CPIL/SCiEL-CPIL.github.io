# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    This deterministic approach calculates the expected amount of improvement
    over the current best observed value f_best for each candidate point.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher values indicate more promising candidates.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the standardized improvement (z-score)
    # alpha = (f_best - mu) / sigma
    # If f_best - mu > 0, there is potential improvement.
    alpha = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(alpha) + sigma * phi(alpha)
    # Where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    # To handle large values of alpha where pdf becomes extremely small and cdf becomes 1,
    # we can use the standard formulation.
    # For very negative alpha (mu >> f_best), EI is ~0.
    # For very positive alpha (mu << f_best), EI is ~ f_best - mu.
    
    # Using scipy.stats.norm for precision
    cdf_alpha = norm.cdf(alpha)
    pdf_alpha = norm.pdf(alpha)
    
    scores = (f_best - mu) * cdf_alpha + sigma_safe * pdf_alpha
    
    # Ensure scores are non-negative (EI should be >= 0)
    # Numerical errors might produce tiny negative values
    scores = np.maximum(scores, 0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check
    elapsed = time.time() - start_time
    if elapsed > 290:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
