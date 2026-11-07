import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    sigma = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma
    # Expected Improvement instead of Probability of Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Numerical stability: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    return ei / cost
