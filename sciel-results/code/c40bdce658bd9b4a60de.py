"""Rewritten acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB).
    
    LCB is effective for exploration in multimodal functions like Goldstein-Price.
    It selects points with low predicted mean and high uncertainty.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Guard against non-finite inputs
    sigma_safe = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma_safe = np.maximum(sigma_safe, 1e-9)
    
    # Lower Confidence Bound for minimization:
    # We want to minimize mu - kappa * sigma.
    # To return "higher is better" scores, we can return -mu + kappa * sigma.
    
    # A dynamic kappa could be used, but a fixed value is a structural change from EI.
    # For Goldstein-Price, significant exploration is needed.
    # Typical LCB kappa ranges from 1 to 5. Let's use 2.0 as a robust default.
    kappa = 2.0
    
    scores = -mu + kappa * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
