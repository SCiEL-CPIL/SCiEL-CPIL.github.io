import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # UCB for minimization: mu - kappa * sigma
    # We want to maximize this score.
    # Cost-aware: divide by cost to penalize expensive points.
    
    # kappa controls exploration. 
    # Since budget is small (~20 evals), we need good exploration.
    # Standard EI is score 1.0. UCB might be better.
    
    kappa = 1.0
    ucb = mu - kappa * sigma
    
    # Avoid division by zero or negative costs (cost is always >= 1)
    # cost is passed as an array of shape (2000,)
    
    # Normalizing ucb might help if scales vary, but mu/sigma are in objective units.
    # Griewank values are around 0-1.
    
    score = ucb / cost
    
    # Ensure finite scores
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
