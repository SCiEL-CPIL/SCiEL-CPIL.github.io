"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to minimize mu - beta * sigma.
    # Therefore, we maximize beta * sigma - mu.
    # We normalize by cost to prefer cheaper evaluations.
    
    # Dynamic beta could be useful, but without iteration count, 
    # a fixed beta is a structural change from EI.
    # Beta = 2.0 is a standard choice for balanced exploration/exploitation.
    beta = 2.0
    
    # Calculate LCB score: higher is better
    # lcb_score = beta * sigma - mu
    # cost_adjusted_score = lcb_score / cost
    
    # Ensure sigma is non-negative and handle any numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    lcb_score = beta * sigma - mu
    
    # Normalize by cost
    # cost is given as an array of shape (2000,)
    # Ensure cost is not zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = lcb_score / cost_safe
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
