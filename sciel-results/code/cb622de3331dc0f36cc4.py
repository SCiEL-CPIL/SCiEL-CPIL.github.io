"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with a dynamic
    improvement threshold (xi) scaled by the current best value (f_best).
    
    Logic:
    - xi is set to a fraction of f_best.
    - This dynamically balances exploration and exploitation:
      - When f_best is large, xi is larger, requiring significant improvement.
      - When f_best is small (close to 0), xi is small, allowing fine-grained improvements.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Ensure f_best is non-negative for scaling, though it should be >= 0 for Weierstrass
    # Weierstrass minimum is 0.
    f_best_safe = max(f_best, 0.0)
    
    # Scale xi by f_best. 
    # A factor of 0.05 provides a moderate threshold.
    # If f_best is 0.5, xi = 0.025.
    # If f_best is 0.01, xi = 0.0005.
    xi = 0.05 * f_best_safe
    
    # Calculate EI with dynamic xi
    # EI_xi = E[max(f_best - f(x) - xi, 0)]
    # z = (f_best - mu - xi) / sigma
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    
    z = (f_best - mu - xi) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    # Although scipy handles them well, clipping to [-10, 10] is safe.
    z = np.clip(z, -10.0, 10.0)
    
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure scores are finite and non-negative
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
