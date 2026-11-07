import numpy as np

def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for minimization.
    # The score is -mu + beta * sigma.
    # We want to maximize this score.
    # A beta of 1.0 provides a balance between exploitation (low mu)
    # and exploration (high sigma).
    beta = 1.0
    return -mu + beta * sigma
