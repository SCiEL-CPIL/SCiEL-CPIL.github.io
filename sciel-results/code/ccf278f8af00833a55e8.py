"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on UCB (Upper Confidence Bound).
    
    Args:
        mu: Array of GP mean predictions, shape (2000,)
        sigma: Array of GP standard deviations, shape (2000,)
        f_best: Best observed function value so far (scalar)
        cost: Array of costs for each candidate, shape (2000,)
        
    Returns:
        Array of acquisition scores, shape (2000,). Higher is better.
    """
    # Exploration parameter
    beta = 2.0
    
    # Ensure cost is not zero to avoid division errors
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB for minimization: mu - beta * sigma
    # We want to maximize the "desirability", so we invert the sign for the score
    # Score = (beta * sigma - mu) / cost
    scores = (beta * sigma - mu) / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
