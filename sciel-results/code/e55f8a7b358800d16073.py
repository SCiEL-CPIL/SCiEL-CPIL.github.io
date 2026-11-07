import numpy as np

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Upper Confidence Bound for minimization
    # We maximize the score which corresponds to the optimistic lower bound:
    # f(x) is assumed to be mu(x) - beta * sigma(x)
    # So we maximize: -mu(x) + beta * sigma(x)
    # beta controls the exploration-exploitation trade-off
    beta = 1.5
    scores = -mu + beta * sigma
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
