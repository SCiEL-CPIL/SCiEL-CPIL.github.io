import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the potential for a low value.
    # The lower bound of the true value is mu - beta * sigma.
    # We want to minimize this lower bound, so we maximize -(mu - beta * sigma) = beta * sigma - mu.
    beta = 2.0
    scores = beta * sigma - mu
    return scores
