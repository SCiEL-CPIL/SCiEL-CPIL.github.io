import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (for minimization, we want low mu and high sigma)
    # Score = -mu + beta * sigma
    beta = 1.0
    ucb = -mu + beta * sigma
    
    # Blend
    score = 0.7 * ei + 0.3 * ucb
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
