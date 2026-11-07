"""Model-and-optimize acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SECONDS = 300.0
_SAFETY_MARGIN = 20.0

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    Score is -mu + kappa * sigma, where higher score means evaluate next.
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Check deadline guard
    if _elapsed() > _DEADLINE_SECONDS - _SAFETY_MARGIN:
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # UCB for minimization:
    # We want to find the minimum.
    # The Lower Confidence Bound (LCB) is mu - kappa * sigma.
    # We want to minimize LCB, which is equivalent to maximizing -LCB.
    # Score = -mu + kappa * sigma
    
    kappa = 1.0
    scores = -mu + kappa * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
