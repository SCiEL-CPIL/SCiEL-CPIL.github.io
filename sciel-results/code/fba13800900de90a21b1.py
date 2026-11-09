"""Rewritten program for cost-aware acquisition in Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 10.0

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        # Fallback: simple greedy exploitation (minimize mu, consider cost)
        mu_arr = np.asarray(mu, dtype=float)
        cost_arr = np.asarray(cost, dtype=float)
        return -mu_arr / (cost_arr + 1e-9)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure sigma is positive to avoid division issues or log issues if used
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive UCB (Upper Confidence Bound) for minimization
    # We want to minimize f(x).
    # The optimistic value (lower bound of prediction) is mu - beta * sigma.
    # We pick the point with the lowest optimistic value.
    # This is equivalent to maximizing the score: (-mu + beta * sigma) / cost.
    
    # Adapt beta based on f_best.
    # If f_best is close to the global minimum (0), we should exploit (lower beta).
    # If f_best is high (stuck in local minimum or unexplored), we should explore (higher beta).
    # Rastrigin 2D has global min 0 and local mins around 8-10.
    # We use a reference scale of 20 to normalize the "badness" of f_best.
    
    fb = max(0.0, float(f_best))
    # Linear interpolation between beta_min and beta_max
    beta_min = 1.0
    beta_max = 4.0
    ref_value = 20.0
    
    # If fb is 0, beta is beta_min. If fb >= ref_value, beta is beta_max.
    beta = beta_min + (beta_max - beta_min) * (fb / ref_value)
    beta = min(beta, beta_max)
    
    # Calculate score
    # Higher score is better
    score = (-mu + beta * sigma) / (cost + 1e-9)
    
    # Ensure finite outputs
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
