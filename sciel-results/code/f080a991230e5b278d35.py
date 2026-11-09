"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track cumulative time spent in acquisition calls
_acq_start_time = [time.time()]
_acq_cumulative_time = [0.0]
DEADLINE_SECONDS = 310.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Griewank 2D.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        GP posterior mean at candidate points
    sigma : ndarray of shape (2000,)
        GP posterior standard deviation at candidate points
    f_best : float
        Lowest observed objective value so far
    cost : ndarray of shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    elapsed = time.time() - _acq_start_time[0] + _acq_cumulative_time[0]
    if elapsed > DEADLINE_SECONDS:
        # Return a simple deterministic score to finish quickly
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    call_start = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma and cost are positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Clip f_best to [0, 1] for stable interpolation
    f_best_clipped = np.clip(f_best, 0.0, 1.0)
    
    # Compute z-score for EI/PI
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Compute standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware EI
    ei_scores = ei / cost
    
    # Cost-aware UCB
    # Adaptive beta: higher when f_best is poor (exploration), lower when good (exploitation)
    beta_min = 0.5
    beta_max = 2.0
    beta = beta_min + (beta_max - beta_min) * f_best_clipped
    ucb = -mu + beta * sigma
    ucb_scores = ucb / cost
    
    # Blend UCB and EI: use more UCB when exploring (high f_best), more EI when exploiting
    # This is the structural idea from inspiration 1 (score 2.0555)
    weight_ucb = f_best_clipped
    weight_ei = 1.0 - f_best_clipped
    
    scores = weight_ucb * ucb_scores + weight_ei * ei_scores
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Update cumulative time
    call_elapsed = time.time() - call_start
    _acq_cumulative_time[0] += call_elapsed
    
    return scores
# EVOLVE-BLOCK-END
