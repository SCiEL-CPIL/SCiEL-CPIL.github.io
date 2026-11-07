import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB (Upper Confidence Bound) for minimization
    # We want to minimize f, so we prefer low mu.
    # The score should be high for promising points.
    # score = -mu + beta * sigma
    beta = 2.0
    return -mu + beta * sigma
