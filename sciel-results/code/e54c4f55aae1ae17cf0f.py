import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # cost is in [1, 2]
    # Normalize cost to [0, 1] where 0 is cheap (1.0) and 1 is expensive (2.0)
    # cost_norm = (cost - 1.0) / 1.0
    
    # We want to encourage exploration in cheap regions more than expensive ones.
    # Let beta be the exploration coefficient.
    # beta_base = 3.0
    # weight = 1.0 - 0.5 * cost_norm  # Range [0.5, 1.0]
    # This reduces exploration weight for expensive points by up to 50%.
    
    # Alternatively, use EI but with a cost-adjusted f_best? No.
    
    # Let's stick to UCB with cost-modulated beta.
    # Score = -mu + beta * sigma
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost modulation
    # cost is shape (2000,)
    # Min cost is 1.0, max is 2.0.
    # Let's map cost to a multiplier in [0.5, 1.0]
    # multiplier = 1.5 - 0.5 * cost
    # If cost=1, mult=1.0
    # If cost=2, mult=0.5
    
    multiplier = 1.5 - 0.5 * cost
    multiplier = np.clip(multiplier, 0.1, 1.0) # Safety
    
    beta = 3.0
    
    score = -mu + beta * multiplier * sigma_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
