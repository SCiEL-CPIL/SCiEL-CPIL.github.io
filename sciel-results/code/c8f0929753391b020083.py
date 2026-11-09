import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound for minimization: maximize -mu + beta * sigma
    # With noise 1e-6 and smooth 1D sphere, strong exploitation helps
    # find the global minimum quickly.
    beta = 0.8
    scores = -mu + beta * sigma
    return scores
