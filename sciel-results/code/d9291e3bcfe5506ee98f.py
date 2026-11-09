"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    mu: (N,) posterior mean
    sigma: (N,) posterior std
    f_best: best observed value so far
    
    Returns: (N,) scores, higher is better.
    """
    # Deadline guard: The computation is O(N) and extremely fast for N=2000.
    # No complex loops or heavy linear algebra are performed, so timeout risk is negligible.
    
    # Clip sigma to avoid division by zero and ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # For minimization, we want to find points where the function value is lower than f_best.
    # Improvement I(x) = max(0, f_best - f(x))
    # Assuming f(x) ~ N(mu(x), sigma(x)^2)
    
    # Standardized improvement:
    # Z = (f_best - mu) / sigma
    # Note: If mu < f_best, Z is positive. Higher Z means higher expected improvement.
    
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # E[I] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF of the standard normal distribution
    # and phi is the PDF of the standard normal distribution.
    
    # Using scipy.stats.norm for accurate CDF and PDF
    from scipy.stats import norm
    
    # Compute Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Calculate EI
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite scores. 
    # If f_best is very low and mu is very high, EI could theoretically be small but positive.
    # If f_best is 0 and mu is 0, EI is 0.
    # We want higher scores to be selected.
    
    # Handle potential numerical issues where EI might be negative due to floating point errors
    # although theoretically EI >= 0.
    ei = np.maximum(ei, 0.0)
    
    # Replace any non-finite values with a very low score (or 0, since EI >= 0)
    # If sigma is huge, EI is large. If sigma is tiny and mu > f_best, EI is 0.
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
