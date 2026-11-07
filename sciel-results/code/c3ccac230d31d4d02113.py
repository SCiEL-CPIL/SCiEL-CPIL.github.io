"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Deterministic Expected Improvement with adaptive exploration parameter xi.
    
    xi scales with the magnitude of f_best to balance exploitation and exploration.
    For minimization, EI is computed on (f_best - f) so that higher scores mean
    more expected improvement over the current best.
    """
    # Adaptive exploration parameter: scales with the absolute value of f_best
    # When f_best is large in magnitude, we explore more; when close to optimum, exploit.
    xi = 0.1 * abs(f_best)
    
    # Guard against zero or near-zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected improvement for minimization:
    # EI(x) = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu - xi) / sigma
    z = (f_best - mu - xi) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Return scores where higher means evaluate next
    return ei
# EVOLVE-BLOCK-END
