import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Ensure sigma is positive to avoid division by zero or log of negative
    # In GP, sigma should be > 0, but numerical errors might make it 0 or negative
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Z-score: (f_best - mu) / sigma
    # If mu < f_best, z is positive. Higher z means more likely to be better.
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We want to maximize this.
    
    # Calculate Phi(z) and phi(z)
    # norm.cdf and norm.pdf are vectorized and efficient
    phi_z = norm.pdf(z)
    phi_cdf = norm.cdf(z)
    
    # EI calculation
    # Note: If sigma is very small and mu < f_best, z is large positive, Phi(z) ~ 1, EI ~ f_best - mu.
    # If mu > f_best, z is negative, Phi(z) is small, EI is small.
    
    ei = (f_best - mu) * phi_cdf + sigma_safe * phi_z
    
    # Handle numerical stability: if sigma is effectively 0, EI should be 0 if mu >= f_best, 
    # and f_best - mu if mu < f_best. 
    # The formula above handles this reasonably well if sigma_safe is not too small.
    # However, if sigma is 0, z is inf. norm.cdf(inf) = 1, norm.pdf(inf) = 0.
    # ei = (f_best - mu) * 1 + 0 = f_best - mu. Correct.
    
    # Replace any non-finite values (should not happen with safe sigma, but just in case)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
