"""Rewritten program using a DIRECT approach for Bayesian optimization acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track global start time for deadline guard
_global_start_time = None
_TIMEOUT_MARGIN = 10.0  # seconds to spare before 330s limit
_TOTAL_LIMIT = 330.0

# Track evaluation count to adjust exploration over time
_eval_count = 0


def _check_deadline():
    """Check if we are approaching the timeout limit."""
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()
    elapsed = time.time() - _global_start_time
    return elapsed > (_TOTAL_LIMIT - _TIMEOUT_MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - lowest observed value
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    global _global_start_time, _eval_count
    
    if _global_start_time is None:
        _global_start_time = time.time()
    
    # Check deadline - if close to timeout, return a safe default
    if _check_deadline():
        # Return Expected Improvement as a safe fallback (fast to compute)
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Increment evaluation counter
    _eval_count += 1
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Adaptive kappa: start high for exploration, decay over time
    # With 30 total evaluations, we want aggressive exploration early
    progress = min(_eval_count / 30.0, 1.0)
    kappa = 3.0 * (1.0 - 0.7 * progress)  # Decays from 3.0 to 0.9
    
    # Upper Confidence Bound (minimization)
    ucb = -mu + kappa * sigma_safe
    
    # Normalize EI
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Normalize UCB
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max > ucb_min:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Adaptive weights: more UCB early, more EI later
    w_ei = 0.3 + 0.5 * progress  # 0.3 -> 0.8
    w_ucb = 0.7 - 0.5 * progress  # 0.7 -> 0.2
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
