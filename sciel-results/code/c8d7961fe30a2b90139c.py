import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Adaptive Upper Confidence Bound (UCB) for minimization.
    
    The exploration parameter beta is adapted based on the current best value f_best.
    When f_best is far from the global minimum (3.0), beta is large to encourage exploration.
    When f_best is close to 3.0, beta is small to encourage exploitation.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    # Ensure sigma is positive to avoid division by zero or negative scores issues
    # Although UCB doesn't divide by sigma, keeping it clean is good practice
    sigma = np.maximum(sigma, 1e-9)
    
    # Global minimum of Goldstein-Price 2D is 3.0
    f_min = 3.0
    
    # Calculate distance from the known global minimum
    # Clip to avoid negative distances if f_best < 3.0 (shouldn't happen, but safe)
    dist = np.maximum(0.0, f_best - f_min)
    
    # Adaptive Beta calculation
    # beta = beta_min + beta_range * (1 - exp(-dist / temperature))
    # As dist -> 0, beta -> beta_min (Exploitation)
    # As dist -> inf, beta -> beta_min + beta_range (Exploration)
    
    beta_min = 0.5
    beta_range = 5.0
    temperature = 10.0
    
    # If dist is very large, exp(-dist/temp) is 0, so factor is 1
    # If dist is small, exp(-dist/temp) is ~1, so factor is ~0
    exploration_factor = 1.0 - np.exp(-dist / temperature)
    beta = beta_min + beta_range * exploration_factor
    
    # UCB for minimization: Score = -mu + beta * sigma
    # Higher score is better (means lower mu and/or higher sigma)
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check if we're running out of time
    if time.time() > deadline:
        # Fallback: Pure exploitation (lowest mean) or simple UCB with fixed beta
        fallback_beta = 1.0
        fallback_scores = -mu + fallback_beta * sigma
        fallback_scores = np.nan_to_num(fallback_scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return fallback_scores
    
    return scores
