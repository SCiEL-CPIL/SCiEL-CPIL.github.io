"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive to avoid division by zero
    # We use a small epsilon to maintain numerical stability in the z-score calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Structural Change: Switch from Expected Improvement (EI) to Upper Confidence Bound (UCB)
    # UCB is a linear combination of the predicted mean and standard deviation.
    # For minimization, we want to maximize the score, so we define:
    # Score = beta * sigma - mu
    # This encourages exploration (high sigma) while exploiting low predicted values (low mu).
    # Unlike EI, UCB does not depend on f_best directly in the formula (except implicitly via the model),
    # which can help escape stagnation where EI might become small or zero.
    
    # Choose a beta parameter. A value of 2.0 is a common heuristic for moderate exploration.
    # We can also make this adaptive, but a fixed value is a simpler structural change.
    # Let's try a slightly higher beta to encourage more exploration in the early stages,
    # or a fixed one. Given the stagnation, we might need more exploration.
    # Let's use a time-decaying beta to start with high exploration and end with low.
    elapsed = time.time() - _start_time
    
    if elapsed < 10.0:
        beta = 3.0
    elif elapsed < 20.0:
        t = (elapsed - 10.0) / 10.0
        beta = 3.0 - 1.0 * t  # 3.0 to 2.0
    else:
        beta = 2.0
    
    ucb_score = beta * sigma_safe - mu
    
    # Ensure all scores are finite
    ucb_score = np.nan_to_num(ucb_score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return ucb_score
# EVOLVE-BLOCK-END
