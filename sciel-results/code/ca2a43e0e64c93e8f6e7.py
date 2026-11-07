"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we're getting close to the 330s limit, return a safe default
    if time.time() - _start_time > 320:
        # Return scores that favor points with low mean (exploitative fallback)
        return -mu

    # Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f, 0)] where f ~ N(mu, sigma^2)
    # For mu < f_best: EI = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu)/sigma
    # For mu >= f_best: EI = 0
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) = CDF, phi(z) = PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI calculation
    ei = np.zeros_like(mu)
    # Only compute EI where mu < f_best (potential improvement)
    mask = mu < f_best
    ei[mask] = (f_best - mu[mask]) * Phi_z[mask] + sigma_safe[mask] * phi_z[mask]
    
    # Ensure finite scores
    ei[~np.isfinite(ei)] = 0.0
    
    return ei
# EVOLVE-BLOCK-END
