import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        sigma = np.maximum(sigma, 1e-9)
        # UCB: (f_best - mu) + beta * sigma
        beta = 1.0
        scores = (f_best - mu + beta * sigma) / cost
        return scores
