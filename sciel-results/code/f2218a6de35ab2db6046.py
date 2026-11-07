import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin for 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: Greedy exploitation
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and cost is positive
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Strategy: Pure Thompson Sampling with adaptive exploration
    # Thompson Sampling has shown to be more effective for escaping local minima
    # in the rosenbrock landscape compared to EI-based methods.
    
    # Adaptive kappa:
    # Early on (high f_best), we want more exploration (higher kappa).
    # Later (low f_best), we want more exploitation (lower kappa).
    
    # f_best=100 -> kappa = 1 + 100/20 = 6.0
    # f_best=10 -> kappa = 1 + 10/20 = 1.5
    # f_best=0 -> kappa = 1.0
    kappa = 1.0 + max(f_best, 0.0) / 20.0
    kappa = np.clip(kappa, 1.0, 8.0)
    
    # Sample from the posterior
    std = kappa * sigma
    y_ts = np.random.normal(mu, std)
    
    # Score: lower y_ts is better, so -y_ts is the "score" (higher is better)
    # Normalize by cost to make it cost-aware
    scores = -y_ts / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
