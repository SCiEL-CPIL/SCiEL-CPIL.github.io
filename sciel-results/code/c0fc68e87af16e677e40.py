import numpy as np

def acquisition(mu, sigma, f_best):
    # GP-UCB for minimization
    # We want to maximize: -mu + beta * sigma
    # beta controls exploration vs exploitation
    # Since we don't have n, we use a fixed beta.
    # A moderate beta like 1.0 or 2.0 is often good.
    # Let's try 1.0 first. If the function is well-scaled, this should be fine.
    # Note: sigma is standard deviation.
    beta = 1.0
    score = -mu + beta * sigma
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
