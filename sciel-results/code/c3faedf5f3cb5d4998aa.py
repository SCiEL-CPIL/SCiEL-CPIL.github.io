import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    EI is a standard acquisition function for Bayesian Optimization that balances
    exploration and exploitation by maximizing the expected improvement over the
    best observed value.
    
    For minimization, the expected improvement is:
    EI(x) = E[max(f_best - f(x), 0)]
    
    Given f(x) ~ N(mu, sigma^2), the closed form is:
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    """
    # Ensure sigma is positive to avoid division by zero or NaNs
    # A small epsilon helps with numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate standard normal CDF (Phi) and PDF (phi)
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Expected Improvement for minimization
    # If mu > f_best, z is negative, Phi(z) is small, (f_best - mu) is negative.
    # The term (f_best - mu) * Phi(z) is negative but small in magnitude?
    # Actually, if mu >> f_best, z is large negative. Phi(z) ~ 0. phi(z) ~ 0. EI ~ 0.
    # If mu < f_best, z is positive. Phi(z) is close to 1. (f_best - mu) is positive.
    # EI ~ (f_best - mu) + small term. This is correct.
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Clip negative values to 0, as EI should be non-negative
    # Numerical errors can cause small negative values
    ei = np.maximum(ei, 0.0)
    
    # Handle any remaining numerical issues
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Return EI as the score (higher is better)
    return ei
