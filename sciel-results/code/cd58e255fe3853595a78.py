import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound for minimization
    # We want to minimize mu + beta * sigma
    # So we maximize -(mu + beta * sigma)
    beta = 2.0
    scores = -(mu + beta * sigma)
    return scores
