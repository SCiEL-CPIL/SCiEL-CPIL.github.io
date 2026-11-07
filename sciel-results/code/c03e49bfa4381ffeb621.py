import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Handle potential division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # Probability of Improvement
    scores = norm.cdf(z)
    return scores
