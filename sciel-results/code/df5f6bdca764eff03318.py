import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    sigma = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    return pi / cost
