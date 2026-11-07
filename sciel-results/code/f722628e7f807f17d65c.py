"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses Expected Improvement (EI) which provides a principled balance
between exploration and exploitation by maximizing the expected
reduction in the objective function value relative to the current best.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points using Expected Improvement.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Deadline safety check
    if time.time() > _DEADLINE - 5.0:
        # Fallback: Simple UCB-like score to ensure return before timeout
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.0 * sigma_safe

    # Ensure sigma is strictly positive to avoid division by zero
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the normalized improvement metric
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, indicating improvement is likely.
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    # Use scipy.stats.norm for numerical stability and speed
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Compute EI
    scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Numerical cleanup: ensure scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
