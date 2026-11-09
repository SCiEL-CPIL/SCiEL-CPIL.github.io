import numpy as np
import time

# We don't need scipy.stats for UCB, just numpy.
# This is faster than computing CDF/PDF.

def acquisition(mu, sigma, f_best, cost):
    # cost is shape (2000,)
    # mu, sigma are shape (2000,)
    
    # Exploration parameter
    # In low-budget BO, high exploration is often needed.
    # Let's try a value that balances exploitation and exploration.
    kappa = 2.5
    
    # Numerator: Potential improvement (optimistic)
    # We want to minimize f. 
    # Optimistic value is mu - kappa * sigma.
    # Improvement over f_best is f_best - (mu - kappa * sigma)
    # = f_best - mu + kappa * sigma
    
    # Denominator: Cost
    # We want to maximize Improvement / Cost
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate score
    # f_best is a scalar
    numerator = f_best - mu + kappa * sigma
    denominator = cost
    
    # Handle division
    # cost is always >= 1, so no div by zero issues.
    scores = numerator / denominator
    
    # Ensure finite
    # If mu or sigma are inf/nan, scores will be inf/nan.
    # The problem says "Return 2000 finite scores".
    # If the GP produces NaNs, we should map them to -inf or a very low number.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
