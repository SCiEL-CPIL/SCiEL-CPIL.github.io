import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # Score = -mu + beta * sigma
    # We maximize this score to minimize the function.
    
    # Adaptive beta based on the number of observations (sigma count)
    # This helps balance exploration and exploitation as the search progresses
    n_obs = len(sigma)
    
    # Start with high exploration, decrease over time
    # For 30 total evaluations, we want to explore early and exploit later
    # beta = 1.0 * np.sqrt(np.log(1 + n_obs))
    
    # A simpler fixed beta that works well for 30 evaluations
    beta = 1.5
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Compute UCB scores
    scores = -mu + beta * sigma_safe
    
    return scores
