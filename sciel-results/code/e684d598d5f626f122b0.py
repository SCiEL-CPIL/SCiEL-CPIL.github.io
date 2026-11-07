"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    # Deadline guard: if we are close to the 300s limit, switch to exploitation
    # to try to squeeze out the last bit of improvement or finish safely.
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Pure exploitation: minimize mu
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)

    # UCB for minimization:
    # We want to maximize the score.
    # Low mu is good. High sigma is good (exploration).
    # Score = -mu + beta * sigma
    
    # Choose beta. 
    # Branin 2D is multi-modal. We need significant exploration.
    # A beta of 2.0 is a common starting point.
    beta = 2.0
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
