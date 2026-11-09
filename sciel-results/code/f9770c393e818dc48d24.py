import numpy as np


def acquisition(mu, sigma, f_best):
       # UCB for minimization: maximize -mu + beta * sigma
       beta = 2.5
       scores = -mu + beta * sigma
       # Ensure finite
       scores[~np.isfinite(scores)] = -np.inf
       return scores
