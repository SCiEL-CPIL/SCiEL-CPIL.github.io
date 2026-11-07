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
    Acquisition function using a hybrid of Expected Improvement (EI) and UCB
    for minimization, with adaptive exploration based on the number of evaluations.
    
    Strategy:
    - Compute EI for minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
      where z = (f_best - mu) / sigma
    - Compute UCB: UCB = -mu + beta * sigma
    - Blend EI and UCB based on an adaptive weight that favors exploration early
      and exploitation later.
    - Use a time-based safety check to return fast greedy scores if near deadline.
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
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # EI for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # UCB for minimization: maximize (-mu + beta * sigma)
    # Adaptive beta: larger early on, smaller later
    # We don't know the exact iteration, but we can use a heuristic based on
    # how close mu is to f_best. If f_best is much lower than typical mu,
    # we're in exploitation mode.
    # For now, use a fixed moderate beta for UCB component
    beta = 0.3
    
    ucb = -mu + beta * sigma_safe
    ucb = np.where(np.isfinite(ucb), ucb, -np.inf)
    
    # Normalize EI and UCB to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    if ei_max - ei_min > 1e-10:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    if ucb_max - ucb_min > 1e-10:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Adaptive weight: favor UCB (exploration) when sigma is large relative to
    # the spread of mu, favor EI (exploitation) when we're close to the optimum.
    # Use the median sigma as a proxy for exploration level
    median_sigma = np.median(sigma_safe)
    mu_range = np.ptp(mu)
    
    if mu_range > 1e-10:
        # High sigma relative to mu range -> more exploration
        exploration_ratio = median_sigma / (mu_range + 1e-10)
    else:
        exploration_ratio = 0.5
    
    # Clamp exploration_ratio to [0, 1]
    exploration_ratio = np.clip(exploration_ratio, 0.0, 1.0)
    
    # Weight for UCB: higher when exploration_ratio is high
    w_ucb = 0.3 * exploration_ratio + 0.1  # Range: [0.1, 0.4]
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
