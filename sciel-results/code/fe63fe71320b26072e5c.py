"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
# Set deadline to 320s to be safe against the 330s kill
_DEADLINE = _START_TIME + 320.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on LCB with cost-scaled exploration.
    
    Parameters:
    -----------
    mu : numpy.ndarray
        GP mean predictions, shape (2000,).
    sigma : numpy.ndarray
        GP standard deviations, shape (2000,).
    f_best : float
        Best observed function value so far.
    cost : numpy.ndarray
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    scores : numpy.ndarray
        Acquisition scores, shape (2000,). Higher is better.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return zeros to indicate no preference, or random? 
        # Zeros are finite and valid.
        return np.zeros_like(mu)

    # Ensure sigma is positive to avoid division by zero
    # Using a small epsilon
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Cost-aware LCB formulation:
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want higher scores for better (lower) LCB values.
    # So score = -(mu - beta * sigma) = beta * sigma - mu
    
    # Scale exploration term by inverse cost to penalize high-cost regions
    # in the uncertainty bonus. Low-cost regions get more exploration.
    cost_safe = np.maximum(cost, eps)
    inv_cost = 1.0 / cost_safe
    
    # Choose beta to balance exploration and exploitation.
    # A moderate value like 1.0-2.0 works well for BO.
    # Since we're scaling by inv_cost (which is in [0.5, 1.0]),
    # we can use a slightly larger base beta.
    beta = 1.5
    
    # Exploration term scaled by inverse cost
    exploration = beta * sigma_safe * inv_cost
    
    # LCB score: higher is better (we negate the minimization objective)
    scores = exploration - mu
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
