import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (PI)
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    # Cost-aware normalization
    return pi / cost
