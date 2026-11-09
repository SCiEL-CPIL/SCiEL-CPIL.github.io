import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound for minimization
    # We want to find the point with the lowest value.
    # The lower confidence bound is mu - beta * sigma.
    # We want to maximize the "optimism" about finding a low value.
    # Optimism = beta * sigma - mu.
    
    # Choose beta. Since targets are standardized in GP but mu/sigma are in objective units,
    # and the function range is ~20 units, sigma is likely ~1-5.
    # beta=2.0 is a reasonable exploration parameter.
    beta = 2.0
    
    # Handle numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Score: Higher is better.
    # We want to maximize (beta * sigma - mu) / cost
    score = (beta * sigma - mu) / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
