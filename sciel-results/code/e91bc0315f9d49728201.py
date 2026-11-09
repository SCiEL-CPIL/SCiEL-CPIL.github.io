import numpy as np

def acquisition(mu, sigma, f_best):
    # Using Upper Confidence Bound (UCB) for minimization.
    # We aim to minimize the function, so we want low mu.
    # We want to explore areas with high uncertainty (high sigma).
    # The score is designed such that higher values are preferred.
    # score = -mu + beta * sigma
    beta = 2.0
    scores = -mu + beta * sigma
    return scores
