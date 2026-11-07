import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Probability of Improvement
    # Structural change from Expected Improvement to Probability of Improvement
    # PI is more robust to GP uncertainty miscalibration and focuses on the likelihood of success
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    return pi / cost
