"""Rewritten acquisition function for Bayesian optimization of 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track the number of evaluations for time-varying beta
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    
    # Deadline guard
    start_time = time.time()
    deadline = 295.0  # seconds, safety margin before 300s kill

    # Ensure inputs are numpy arrays with correct dtype
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Increment call count (approximation of t)
    _call_count += 1
    t = _call_count

    # Prevent division by zero and negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)

    # Strategy: GP-UCB with time-varying beta
    # beta_t decreases relative to the scale of the problem as we gain more information
    # Using a log-based decay to start with high exploration and settle into exploitation
    # beta_t = sqrt(log(t+1)) is standard, but we might need a larger initial beta
    # Let's use beta_t = sqrt(log(t+1)) + 1.0 to ensure early exploration
    
    beta_t = np.sqrt(np.log(t + 1)) + 1.0
    
    # GP-UCB for minimization: -mu + beta * sigma
    # We want to maximize this score
    ucb_score = -mu + beta_t * sigma_safe

    # Check deadline
    if time.time() - start_time > deadline - 10:
        # Fallback to EI if we are running out of time, though UCB is faster
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei

    # Ensure all scores are finite
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)

    # Final deadline check
    if time.time() - start_time > deadline - 5:
        return ucb_score

    return ucb_score
# EVOLVE-BLOCK-END
