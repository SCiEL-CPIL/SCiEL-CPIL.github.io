"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive to avoid division by zero
    # We use a small epsilon to maintain numerical stability in the z-score calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the PDF.
    # This formula gives the expected amount by which f_best will be improved.
    
    # Compute CDF and PDF
    # scipy.stats functions are vectorized
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Numerical safeguard:
    # When sigma is very small, the continuous EI formula can suffer from floating point cancellation
    # or behave unstably. In the limit sigma -> 0, EI(x) -> max(0, f_best - mu(x)).
    # We apply this piecewise for very small sigma to ensure stability.
    small_sigma_mask = sigma < 1e-8
    if np.any(small_sigma_mask):
        ei[small_sigma_mask] = np.maximum(0, f_best - mu[small_sigma_mask])
        
    # Add exploration bonus to encourage visiting high-uncertainty regions
    # This helps escape local optima in multimodal functions like Hartmann-6
    ei += 0.1 * sigma_safe
    
    # Ensure all scores are finite
    # EI is theoretically non-negative, but numerical errors might cause tiny negatives or infs
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=0.0)
    # Clip negatives to 0 as EI cannot be negative
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
