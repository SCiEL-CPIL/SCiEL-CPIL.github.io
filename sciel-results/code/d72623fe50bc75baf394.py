"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: kill if we exceed ~325s total (safety margin for 330s limit)
_DEADLINE = time.time() + 325.0


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    EI(x) = E[max(f_best - f(x), 0)]
         = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    
    where z = (f_best - mu(x)) / sigma(x),
          Phi is the standard normal CDF,
          phi is the standard normal PDF.
    
    This is a more principled acquisition function than PI as it accounts for
    the magnitude of potential improvement, not just the probability.
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    if time.time() > _DEADLINE - 5.0:
        return -mu + 0.01 * sigma
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score: how many standard deviations f_best is above mu
    z = (f_best - mu) / sigma_safe
    
    # Compute EI:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # When z is large positive (f_best >> mu), EI ≈ f_best - mu
    # When z is large negative (f_best << mu), EI ≈ 0
    # When z ≈ 0, EI ≈ sigma * phi(0) = sigma / sqrt(2*pi)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei_scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure non-negative (EI should always be >= 0)
    ei_scores = np.maximum(ei_scores, 0.0)
    
    # Ensure all values are finite
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return ei_scores
# EVOLVE-BLOCK-END
