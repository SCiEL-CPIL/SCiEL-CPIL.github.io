"""Rewritten acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) with a small exploration bonus.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean predictions at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is positive when mu < f_best (improvement expected)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip to non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finiteness
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
