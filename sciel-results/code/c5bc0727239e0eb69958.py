# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Upper Confidence Bound (UCB) divided by cost.
    UCB balances exploration and exploitation using a beta parameter.
    Score = (mu + beta * sigma) / cost.
    """
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Beta parameter for exploration
    # A higher beta encourages more exploration. 
    # 2.0 is a standard choice, but we can try 2.5 for more aggressive exploration
    # in a constrained budget.
    beta = 2.5
    
    # Calculate UCB score
    ucb = mu + beta * sigma_safe
    
    # Cost-aware score: UCB per unit cost
    # Higher score is better
    score = ucb / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
