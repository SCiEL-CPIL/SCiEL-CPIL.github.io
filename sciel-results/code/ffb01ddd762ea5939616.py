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
    
    Uses a combination of Expected Improvement (EI) and a diversity/uncertainty
    bonus to balance exploitation and exploration. The Branin function has
    multiple local minima, so exploration is important.
    
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
        # Fallback: pure EI with small epsilon
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to [0, 1] range for mixing
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Compute uncertainty (standard deviation) - useful for exploration
    # Normalize sigma to [0, 1]
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Use a simpler and more robust approach:
    # Blend EI with a UCB term for exploration
    
    # Compute UCB with a fixed beta that balances exploration
    # For Branin, beta around 1.0-2.0 works well
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Normalize UCB to [0, 1]
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Mix EI and UCB: 0.8 EI + 0.2 UCB
    # Scale UCB to EI's scale
    scale = max(ei_max, 1e-9)
    
    score = 0.8 * ei + 0.2 * ucb_norm * scale
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
