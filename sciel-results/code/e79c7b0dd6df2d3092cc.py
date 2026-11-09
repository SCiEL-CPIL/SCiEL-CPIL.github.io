"""Acquisition function for Bayesian optimization of beale 2d using Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special as sp

_start_time = time.time()
TIME_LIMIT = 300.0  # Safety margin from 330s

_acq_call_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling with a time-adaptive bias
    toward exploitation in later stages.
    
    Thompson Sampling draws a single sample from the posterior predictive
    distribution at all candidates and returns these samples as scores.
    This naturally balances exploration and exploitation without manual
    tuning of exploration-exploitation tradeoff parameters.
    
    Parameters:
    mu (np.array): Mean of the GP posterior, shape (2000,)
    sigma (np.array): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the 2000 candidates, shape (2000,). Higher is better.
    """
    global _acq_call_count
    _acq_call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to exploitation
        return -np.asarray(mu, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Calculate progress (0 to 1)
    step = (_acq_call_count - 1) % 25
    progress = step / 25.0
    
    # Thompson Sampling: draw a single sample from the posterior predictive
    # The sample is mu - sigma * z where z ~ N(0, 1)
    # Since we want to minimize, we want high scores for low function values
    # So we return the negative of the sampled function value
    z = np.random.standard_normal(len(mu))
    sample = mu - sigma * z
    
    # In later stages, blend toward exploitation (use mu directly)
    # This ensures convergence to the best region
    exploit_weight = progress ** 2  # Quadratic increase in exploitation
    
    # Blend Thompson sample with exploitation
    # When exploit_weight is 0, we use pure Thompson Sampling
    # When exploit_weight is 1, we use pure exploitation (-mu)
    # We want higher scores for lower function values
    score = (1.0 - exploit_weight) * (-sample) + exploit_weight * (-mu)
    
    # Ensure scores are finite
    score[~np.isfinite(score)] = 0.0
    
    return score
# EVOLVE-BLOCK-END
