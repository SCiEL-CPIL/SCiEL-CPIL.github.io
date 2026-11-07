import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Args:
        mu: Array of posterior means (2000,)
        sigma: Array of posterior standard deviations (2000,)
        f_best: Current best observed value (scalar)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Avoid division by zero and negative sigma
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Z-score for improvement: Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Compute Phi(Z) and phi(Z)
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    # EI calculation
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure non-negative (EI should be >= 0)
    EI = np.maximum(EI, 0.0)
    
    # Ensure finite values
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    return EI
