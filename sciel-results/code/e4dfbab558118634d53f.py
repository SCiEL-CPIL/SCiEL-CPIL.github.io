"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: return early if we're getting close to the 330s wall-clock limit
_start_time = time.time()
_DEADLINE_MARGIN = 15.0  # seconds of safety margin
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using an adaptive hybrid of EI and UCB.
    
    The beta parameter for UCB decays with the number of evaluations,
    encouraging more exploration early and more exploitation later.
    The blend between EI and UCB also adapts: early on we favor UCB
    for broad exploration, later we favor EI for focused exploitation.
    
    Parameters:
    mu: array-like, shape (N,), mean of the posterior distribution
    sigma: array-like, shape (N,), standard deviation of the posterior distribution
    f_best: float, the best (lowest) observed function value so far
    
    Returns:
    scores: numpy array, shape (N,), higher values indicate better candidates
    """
    global _call_count
    
    # Check deadline
    if time.time() - _start_time > 330.0 - _DEADLINE_MARGIN:
        # Return a simple UCB score as fallback
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, f_best)
        sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
        sigma = np.maximum(sigma, 1e-9)
        scores = -mu + 2.0 * sigma
        return np.where(np.isfinite(scores), scores, 0.0)
    
    _call_count += 1
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive parameters based on evaluation count
    # We have 30 total evaluations (5 initial + 25 BO steps)
    # _call_count goes from 1 to 25 for the BO steps
    # Progress: 0 at first BO step, 1 at last BO step
    n_total = 25  # number of BO acquisition calls expected
    progress = min(_call_count / n_total, 1.0)
    
    # Beta decays from 2.5 (early, high exploration) to 1.0 (late, exploitation)
    beta = 2.5 - 1.5 * progress
    
    # Blend weight: early on favor UCB (w_ucb high), later favor EI
    # w_ei goes from 0.3 to 0.7
    w_ei = 0.3 + 0.4 * progress
    w_ucb = 1.0 - w_ei
    
    # Calculate Z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    ei = (f_best - mu) * cdf_val + sigma * pdf_val
    
    # UCB component: -mu + beta * sigma
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB with adaptive weights
    scores = w_ei * ei + w_ucb * ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
