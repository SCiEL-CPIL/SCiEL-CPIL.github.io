import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We use the Lower Confidence Bound (LCB) for the objective value.
    # LCB = mu - beta * sigma
    # We want to select the point with the lowest LCB.
    # Since higher scores are selected, we return -LCB.
    # Score = -mu + beta * sigma
    
    beta = 2.0
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
