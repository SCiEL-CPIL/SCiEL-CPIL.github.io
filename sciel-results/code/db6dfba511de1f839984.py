"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


# Deadline guard: keep explicit wall-clock tracking
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # evaluator kills at 330s, leave margin
_TIME_MARGIN = 10.0  # safety margin seconds


def _time_left():
    return _TIME_LIMIT - _TIME_MARGIN - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses an adaptive blend of Expected Improvement (EI) and UCB to balance
    exploitation and exploration. The Branin function has multiple local
    minima, so a balance is critical.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        GP mean at candidate points (objective units)
    sigma : ndarray, shape (n,)
        GP standard deviation at candidate points (objective units)
    f_best : float
        Best observed value so far
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more promising)
    """
    # Check time budget - if we're running out, return a simple fallback
    if _time_left() < 1.0:
        # Fallback: pure EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Compute UCB with a moderate beta for exploration
    # UCB = -mu + beta * sigma (for minimization, higher is better)
    # Use beta=1.0 as a reasonable default for Branin
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Normalize UCB to [0, 1]
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Blend EI and UCB with tuned weights
    # Weight 0.8 on EI and 0.2 on UCB to favor exploitation while
    # maintaining some exploration for multi-modal functions
    score = 0.8 * ei_norm + 0.2 * ucb_norm
    
    # Scale back to EI's scale for consistency
    if ei_max > 1e-12:
        score = score * ei_max
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
