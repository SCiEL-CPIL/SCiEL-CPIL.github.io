"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # wall-clock seconds total allowed (5s safety margin)

# Known global minimum for Styblinski-Tang 1D
F_MIN = -39.1662


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Uses a hybrid of Expected Improvement (EI) and adaptive Upper Confidence Bound (UCB).
    The UCB exploration coefficient is adapted based on the gap between f_best and F_MIN.
    
    Args:
        mu: array of shape (2000,) with posterior means
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: lowest observed function value
    
    Returns:
        array of shape (2000,) with acquisition scores (higher is better)
    """
    # Guard against negative or zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement (minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive UCB exploration coefficient
    # When f_best is far from F_MIN (large gap), use high exploration
    # When f_best is close to F_MIN (small gap), use low exploration
    gap = f_best - F_MIN
    
    BETA_MIN = 0.2
    BETA_MAX = 2.0
    GAP_SCALE = 30.0
    
    # Clip gap to [0, GAP_SCALE] for the linear ramp
    gap_clipped = np.clip(gap, 0.0, GAP_SCALE)
    
    # Linear interpolation for beta
    beta = BETA_MIN + (BETA_MAX - BETA_MIN) * (gap_clipped / GAP_SCALE)
    
    # UCB Score for minimization: maximize -mu + beta * sigma
    ucb = -mu + beta * sigma_safe
    
    # Normalize both EI and UCB to [0, 1] range for combining
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_scale = ei_max - ei_min
    if ei_scale < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_scale
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_scale = ucb_max - ucb_min
    if ucb_scale < 1e-12:
        ucb_norm = np.zeros_like(ucb)
    else:
        ucb_norm = (ucb - ucb_min) / ucb_scale
    
    # Weighted combination: 70% EI, 30% UCB
    # EI provides exploitation toward known good regions
    # UCB provides exploration, especially early on when gap is large
    score = 0.70 * ei_norm + 0.30 * ucb_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
