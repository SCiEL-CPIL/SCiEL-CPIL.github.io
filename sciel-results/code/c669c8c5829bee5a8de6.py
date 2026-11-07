import numpy as np
from scipy.special import erfc

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    Computes the exact expected improvement of each candidate over f_best.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Floor sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score: how many standard deviations f_best is above the mean
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical overflow in erfc
    z_clipped = np.clip(z, -10, 10)
    
    # Compute standard normal CDF and PDF
    sqrt_2 = np.sqrt(2.0)
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    
    # Phi(z) = 0.5 * erfc(-z / sqrt(2))
    Phi = 0.5 * erfc(-z_clipped / sqrt_2)
    
    # phi(z) = exp(-z^2/2) / sqrt(2*pi)
    phi = np.exp(-0.5 * z_clipped**2) / sqrt_2pi
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # EI is non-negative by construction, but clip for safety
    EI = np.maximum(EI, 0.0)
    
    # Ensure finite scores
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    return EI
