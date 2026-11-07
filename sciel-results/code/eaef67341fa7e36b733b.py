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
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Z-score for improvement
    # We want f(x) < f_best
    # Let Y = f(x). Y ~ N(mu, sigma^2)
    # Improvement I = max(0, f_best - Y)
    # E[I] = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    # Expected Improvement
    EI = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Ensure finite values
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    return EI
