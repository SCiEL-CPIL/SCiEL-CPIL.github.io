import numpy as np
from scipy.stats import norm
import time

# Track call count and time for safety
_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Shekel 4d.
    
    Uses Thompson Sampling: sample a function value from the GP posterior
    f(x) ~ N(mu, sigma^2), then score by -sampled_value / cost.
    This naturally balances exploration and exploitation without tuning
    hyperparameters. The negative sign ensures higher scores correspond
    to lower (better) function values.
    """
    global _call_count, _start_time
    
    # Initialize timing on first call
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we've been running too long, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 300:  # Safety margin before 330s limit
        return -mu / cost
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: draw one sample from the GP posterior for each candidate
    # z ~ N(0, 1)
    z = np.random.standard_normal(len(mu))
    sampled_values = mu + sigma_safe * z
    
    # Score: we want to minimize sampled_value / cost, so negate for maximization
    score = -sampled_values / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
