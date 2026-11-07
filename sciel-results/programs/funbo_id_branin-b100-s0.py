"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses a combination of Expected Improvement (EI) and a UCB-like exploration term
    to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate point. Higher means evaluate next.
    """
    start_time = time.time()
    
    # Safety margin: ensure we finish well before the 300s limit
    DEADLINE = 300.0
    SAFETY_MARGIN = 5.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    # UCB-style exploration term
    # Balance exploitation (via EI) and exploration (via sigma)
    # The beta parameter controls exploration; for 2D problems, ~1.5-2.5 works well
    beta = 1.8
    ucb = -mu + beta * sigma_safe
    
    # Blend EI and UCB: normalize both to comparable scales
    # Use a weighted combination that adapts based on the current state
    # When f_best is far from typical values, favor exploration more
    
    # Normalize EI and UCB to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    ei_range = ei_max - ei_min + eps
    ucb_range = ucb_max - ucb_min + eps
    
    ei_norm = (ei - ei_min) / ei_range
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Weight: more weight to EI as we get closer to the optimum
    # Use a simple heuristic: if f_best is small (close to optimum), favor exploitation
    # Branin minimum is ~0.397887, so if f_best is close, increase EI weight
    if f_best < 1.0:
        w_ei = 0.7
    elif f_best < 3.0:
        w_ei = 0.5
    else:
        w_ei = 0.3
    
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        # Return current scores; they should already be computed
        pass
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
