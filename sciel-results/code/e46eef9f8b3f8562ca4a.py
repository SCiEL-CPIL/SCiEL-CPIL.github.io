import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (PI) for minimization
    # PI(x) = P(Y(x) < f_best) = Phi((f_best - mu) / sigma)
    
    # Handle zero sigma
    sigma_safe = np.where(sigma < 1e-10, 1e-10, sigma)
    
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Cost-aware score: PI / cost
    scores = pi / cost
    
    return scores
