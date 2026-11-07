import numpy as np

def acquisition(mu, sigma, f_best):
    beta = 2.0
    # Minimization: we want low mu.
    # UCB for minimization is often defined as mu - beta * sigma (lower bound).
    # We want to maximize the score.
    # So we maximize -(mu - beta * sigma) = -mu + beta * sigma.
    return -mu + beta * sigma
