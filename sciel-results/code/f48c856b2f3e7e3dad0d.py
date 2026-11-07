import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB)
    # We want to maximize the acquisition function.
    # Lower mu is better (exploitation).
    # Higher sigma is better (exploration).
    beta = 2.0
    return -mu + beta * sigma
