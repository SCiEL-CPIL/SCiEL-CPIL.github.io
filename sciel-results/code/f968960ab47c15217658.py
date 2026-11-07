"""Rewritten program: direct cost-aware acquisition for Rastrigin 2D Bayesian optimization."""
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
    
    # Cost is in [1, 2] (1 + mean of x scaled to [0,1])
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Upper Confidence Bound for minimization
    # UCB = -mu + kappa * sigma
    # We maximize this score.
    # Cost-aware: divide by cost to prefer cheaper points.
    
    # Dynamic kappa: start high for exploration, decrease slightly?
    # Or keep constant. Let's try a constant kappa first.
    # Typical values for UCB in BO are 1-5.
    kappa = 2.0
    
    ucb = -mu + kappa * sigma
    
    # Cost-aware normalization
    # Using 1/cost instead of 1/cost^2 to be less aggressive on cost
    # This allows exploring high-cost regions if the potential improvement is high enough
    score = ucb / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
