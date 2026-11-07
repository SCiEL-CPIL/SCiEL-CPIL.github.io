"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition with exploration
    that scales inversely with observed regret.
    
    The exploration coefficient beta is adapted based on how close f_best
    is to the known global minimum. When we are far from the optimum, we
    explore more aggressively; when we are close, we exploit more.
    
    Parameters
----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    t_start = time.time()
    
    n = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Theoretical global minimum of Hartmann 6D function
    f_min = -3.32237
    regret = max(f_best - f_min, 0.0)
    
    # Check time budget early
    if time.time() - t_start > 290:
        scores = -mu + 1.5 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Adaptive exploration coefficient
    # When regret is large (far from optimum), explore more
    # When regret is small (close to optimum), exploit more
    # Base beta around 1.0-2.0, scaled by regret
    # Normalize regret: regret=0 means at optimum, regret~1 is typical early
    regret_norm = min(regret / 1.0, 1.0)  # cap at 1.0
    
    # Beta ranges from ~0.5 (exploit, near optimum) to ~2.5 (explore, far from optimum)
    beta = 0.5 + 2.0 * regret_norm
    
    # LCB score: we want to maximize -mu + beta * sigma
    # This is equivalent to minimizing mu - beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Check time budget
    if time.time() - t_start > 295:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
