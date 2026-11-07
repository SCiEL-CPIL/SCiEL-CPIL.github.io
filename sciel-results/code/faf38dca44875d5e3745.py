import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.5, posinf=2.0, neginf=1.0)
    
    # Cost is in [1, 2]
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Structural Change: Use Upper Confidence Bound (UCB) instead of Expected Improvement (EI)
    # UCB = -mu + kappa * sigma
    # This promotes exploration more aggressively than EI, which is purely exploitation-driven.
    kappa = 2.0
    ucb = -mu + kappa * sigma
    
    # Cost-aware Score: UCB / Cost
    # Linear cost penalty (vs quadratic in previous version)
    # This balances exploration/exploitation with cost efficiency
    score = ucb / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
