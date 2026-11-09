import numpy as np
from scipy.stats import norm
import time

# Deadline guard: kill safety margin
_DEADLINE = time.time() + 320.0

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Confidence Bound (CB) with adaptive beta
    # Objective: Minimize f(x)
    # Lower Bound: L(x) = mu(x) - beta * sigma(x)
    # Acquisition: Maximize (-L(x)) / cost(x)
    
    # Check deadline - if close to limit, return safe scores quickly
    if time.time() > _DEADLINE - 10:
        # Emergency: return simple scores
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        scores = (-mu + 1.0 * sigma_safe) / cost_safe
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Adaptive beta based on convergence state
    # When f_best is close to 0 (converged), reduce exploration
    # When f_best is far from 0, increase exploration
    # Normalize f_best: assume typical range is [0, ~10] for three hump camel
    # f_best starts high and decreases toward 0
    
    # Use a smooth transition for beta
    # beta_high = 3.0 (early, high uncertainty)
    # beta_low = 1.5 (late, converged)
    # Transition based on f_best: when f_best < 0.5, we're close to optimum
    
    f_best_val = float(f_best)
    
    # Smooth transition: beta = beta_low + (beta_high - beta_low) * sigmoid(-k * (f_best - threshold))
    # Use threshold = 0.5, k = 5
    threshold = 0.5
    k = 5.0
    # sigmoid(-k * (f_best - threshold)) = 1 / (1 + exp(k * (f_best - threshold)))
    arg = k * (f_best_val - threshold)
    # Clamp arg to avoid overflow
    arg = np.clip(arg, -50, 50)
    sigmoid_val = 1.0 / (1.0 + np.exp(arg))
    
    beta_high = 3.0
    beta_low = 1.5
    beta = beta_low + (beta_high - beta_low) * sigmoid_val
    
    # Calculate the score
    # -mu + beta * sigma
    numerator = -mu + beta * sigma
    scores = numerator / cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
