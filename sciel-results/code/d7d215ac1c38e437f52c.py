"""Acquisition function for Bayesian optimization of a 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: keep a global start time and a safety margin.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 325s out of 330s budget

def _time_remaining():
    return _DEADLINE - time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining expected improvement with a
    diversity-aware exploration bonus.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP posterior std at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising points.
    """
    # Check deadline
    if _time_remaining() < 0.5:
        # Return a simple EI as fallback to ensure we finish in time
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Core: Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty to avoid
    # over-concentrating on the current best region.
    # Scale the bonus so it's meaningful relative to EI values.
    # The bonus encourages visiting high-uncertainty areas.
    # We use a tempered version: bonus = alpha * sigma / (1 + |mu - f_best|/sigma)
    # This gives more bonus when uncertainty is high relative to distance from best.
    
    # Normalize EI to have comparable scale with the bonus
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # Exploration bonus: reward high sigma, but dampen if mu is already very low
    # (i.e., don't explore where we already have good predictions)
    # bonus = sigma / (1 + np.abs(f_best - mu) / (sigma + 1e-9))
    # This is bounded by sigma, so scale it down.
    
    # A simpler approach: weighted combination of EI and sigma
    # The weight on sigma should be small to not dominate EI
    # Use a fixed small weight
    w_explore = 0.05
    
    scores = ei_norm + w_explore * (sigma / (sigma_safe.max() + 1e-9))
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
