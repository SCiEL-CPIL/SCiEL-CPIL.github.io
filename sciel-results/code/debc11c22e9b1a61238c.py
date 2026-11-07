"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track wall-clock deadline
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to leave before hard stop

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Strategy:
    - Use UCB: score = -mu + beta * sigma
    - This explicitly balances exploitation (-mu) and exploration (sigma).
    - Beta is chosen to be moderate to avoid excessive exploration in a small domain.
    """
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    # Safety margin: if we are close to the deadline, return a fast greedy score
    if remaining_time < _DEADLINE_MARGIN:
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores

    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # UCB for minimization: we want high scores for low mu and high sigma.
    # Standard UCB for maximization is mu + beta * sigma.
    # For minimization, we want to maximize (-mu + beta * sigma).
    beta = 0.5
    
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
