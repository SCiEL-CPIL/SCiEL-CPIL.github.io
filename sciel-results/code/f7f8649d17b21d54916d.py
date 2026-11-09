import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    Score = -mu + beta * sigma.
    Higher score indicates a more promising candidate.
    """
    beta = 2.0
    return -mu + beta * sigma
