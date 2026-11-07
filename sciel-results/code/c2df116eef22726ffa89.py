"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization.
    # We want to maximize the score, which is the expected improvement.
    # Improvement = f_best - f(x). We want to maximize E[max(f_best - f(x), 0)].
    
    # Handle zero or near-zero sigma to avoid division by zero.
    # If sigma is very small, the function value is almost certain.
    # If mu < f_best, EI is (f_best - mu).
    # If mu >= f_best, EI is 0 (approximately, if sigma is 0).
    
    sigma_safe = np.where(sigma > 1e-8, sigma, 1.0)
    
    # Z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # PDF and CDF of standard normal
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    # EI formula
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # If sigma was effectively 0, we should just take the deterministic improvement if positive, else 0.
    # However, the standard formula handles small sigma well if we mask it.
    # For sigma == 0:
    # If mu < f_best, improvement is f_best - mu.
    # If mu >= f_best, improvement is 0.
    
    # Let's refine for exact zero sigma cases just in case
    mask_zero_sigma = sigma <= 1e-8
    if np.any(mask_zero_sigma):
        deterministic_ei = np.maximum(f_best - mu[mask_zero_sigma], 0)
        ei[mask_zero_sigma] = deterministic_ei
        
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
