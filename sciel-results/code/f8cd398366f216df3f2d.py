import numpy as np
from scipy.stats import norm
import time

# Global time tracking for safety
_start_time = time.time()
_time_limit = 300.0 # Seconds

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are dangerously close to the limit, return a safe fallback
    if time.time() - _start_time > _time_limit - 5.0:
        return np.zeros_like(mu)
    
    # Safe sigma to avoid division by zero
    safe_sigma = np.maximum(sigma, 1e-9)
    
    # UCB Calculation
    # UCB(x) = mu(x) + beta * sigma(x)
    # Since we're minimizing, we want to find points with low mu and high uncertainty
    # The acquisition should be higher for points likely to improve f_best
    # For minimization: improvement potential = f_best - mu(x)
    # Exploration bonus = beta * sigma(x)
    # So score = (f_best - mu) + beta * sigma
    
    beta = 2.0  # Exploration constant
    
    # Base improvement potential
    improvement = f_best - mu
    
    # Exploration bonus
    exploration = beta * safe_sigma
    
    # Combined UCB-style score (higher is better for evaluation)
    ucb_score = improvement + exploration
    
    # Cost-aware score with linear penalty
    safe_cost = np.maximum(cost, 1e-9)
    score = ucb_score / safe_cost
    
    # Handle any numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
