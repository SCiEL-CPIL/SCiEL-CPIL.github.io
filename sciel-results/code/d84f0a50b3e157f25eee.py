"""
Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture.
Uses Expected Improvement (EI) with numerical stability refinements.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 330.0  # 330 seconds wall-clock limit
_SAFETY_MARGIN = 5.0 # Stop 5 seconds before the limit

def _check_deadline():
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) acquisition score for minimization.
    
    EI(x) = E[max(f_best - f(x), 0)]
    
    Returns higher scores for more promising candidates.
    """
    # Check if we are close to the time limit
    # If so, return a safe, quick result (e.g., pure exploitation or random)
    # However, EI is very fast, so this is unlikely to trigger for n=2000.
    if _check_deadline():
        # Fallback: return -mu (pure exploitation) or just mu (if maximization, but we minimize)
        # Since we minimize, lower mu is better. But EI is standard.
        # To be safe and fast, just return a constant or simple metric.
        # Actually, if we are out of time, we should probably return something that doesn't crash.
        # Returning 0.0 is safe but might stop progress.
        # Returning -mu encourages picking the best mean.
        return -mu

    # Ensure sigma is not zero to avoid division by zero
    # Use a small floor for numerical stability
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values where erf/exp might lose precision or become unstable
    # erf(10) is essentially 1.0, exp(-50) is essentially 0.0
    z_clipped = np.clip(z, -10, 10)
    
    # Compute Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # Compute phi(z) = exp(-z^2 / 2) / sqrt(2*pi)
    sqrt2 = np.sqrt(2.0)
    cdf = 0.5 * (1.0 + erf(z_clipped / sqrt2))
    pdf = np.exp(-0.5 * z_clipped * z_clipped) / np.sqrt(2.0 * np.pi)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure non-negative (numerical safety, though EI should be >= 0 analytically)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
