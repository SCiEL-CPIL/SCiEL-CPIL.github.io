import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    eps = 1e-9
    sigma = np.maximum(sigma, eps)
    
    # Probability of Improvement (PI)
    # We want to maximize P(f(x) < f_best)
    # For a Gaussian posterior N(mu, sigma^2),
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    Z = (f_best - mu) / sigma
    scores = norm.cdf(Z)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
