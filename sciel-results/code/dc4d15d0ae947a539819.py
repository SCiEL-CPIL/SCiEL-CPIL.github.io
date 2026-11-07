import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Use LCB for minimization
    # Maximize beta * sigma - mu
    beta = 2.0
    # Normalize by sqrt(cost) to reduce bias towards cheap points
    # but still account for cost
    score = (beta * sigma - mu) / np.sqrt(cost)
    return score
