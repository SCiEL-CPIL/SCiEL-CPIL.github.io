import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) acquisition function for minimization.
    # We want to minimize the function, so we look for low mu and high sigma.
    # LCB(x) = mu(x) - beta * sigma(x)
    # We return a score where higher is better, so we negate the LCB.
    # Score(x) = -mu(x) + beta * sigma(x)
    
    # Beta controls the exploration-exploitation trade-off.
    # A value of 1.0 is a standard choice.
    beta = 1.0
    
    # Ensure sigma is non-negative and handle potential numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
