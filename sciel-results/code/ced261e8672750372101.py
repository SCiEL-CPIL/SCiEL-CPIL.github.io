"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guarding
_START_TIME = time.time()
# Safety margin in seconds (300s limit, leave 10s buffer)
_DEADLINE_MARGIN = 290.0


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: Expected Improvement (EI) with a small exploration bonus.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition values; higher means evaluate next.
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_MARGIN:
        return -mu

    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma^2)
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -8.0, 8.0)
    
    ei = improvement * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Add a small exploration bonus to encourage visiting uncertain regions
    # This helps escape local optima in the multimodal Weierstrass function
    # Use a small coefficient to not overwhelm the exploitation signal
    exploration_weight = 0.1
    scores = ei + exploration_weight * sigma
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
