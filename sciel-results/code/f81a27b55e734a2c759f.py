"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Computes cost-aware Upper Confidence Bound (UCB).
    
    Args:
        mu: array of shape (N,) predicted mean
        sigma: array of shape (N,) predicted std deviation
        f_best: float, best value found so far
        cost: array of shape (N,) evaluation cost for each candidate
        
    Returns:
        scores: array of shape (N,) acquisition values (higher is better)
    """
    # Deadline guard
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to a small positive value to prevent numerical issues
    sigma_clipped = np.maximum(sigma, 1e-9)
    
    # Clip cost to a small positive value
    cost_clipped = np.maximum(cost, 1e-9)
    
    # UCB parameter. 
    # For minimization, we want to maximize (beta * sigma - mu).
    # beta controls the exploration/exploitation trade-off.
    # A value around 2.0-5.0 is typical. 
    # We can also scale beta based on the number of observations if we had that info, 
    # but here we only have f_best. 
    # Let's use a fixed beta of 3.0 which is a reasonable default for multimodal functions.
    beta = 3.0
    
    # Calculate UCB score:
    # We want to find low values, so we subtract the mean.
    # We add the uncertainty (sigma) scaled by beta to encourage exploration.
    ucb = beta * sigma_clipped - mu
    
    # Cost-aware score
    # We divide by cost to prefer cheaper evaluations.
    scores = ucb / cost_clipped
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
# EVOLVE-BLOCK-END
