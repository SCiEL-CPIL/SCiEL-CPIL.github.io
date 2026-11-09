import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    scores = pi / cost
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores
