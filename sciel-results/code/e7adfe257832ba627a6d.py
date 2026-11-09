"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using adaptive Lower Confidence Bound (LCB).
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at 2000 candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at 2000 candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    start_time = time.time()
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive exploration weight: larger when f_best is far from optimum (0),
    # smaller when f_best is close. Beale minimum is 0, max is ~60.
    # Use a smooth decay: weight = base * (1 + f_best / scale) clipped
    f_best_clipped = max(f_best, 1e-6)
    scale = 10.0  # characteristic scale for Beale function values
    exploration_weight = 0.15 * (1.0 + f_best_clipped / scale)
    
    # LCB: minimize mu - beta * sigma, so maximize -(mu - beta * sigma) = -mu + beta * sigma
    # Return negative LCB so that higher score means more promising
    scores = -mu + exploration_weight * sigma_safe
    
    # Safety margin: if we're running out of time, just return quickly
    elapsed = time.time() - start_time
    if elapsed > 250:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
