import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB for minimization: maximize (beta * sigma - mu)
    # We want to find the minimum, so we look for low mu and high sigma.
    # The "optimistic" value is mu - beta * sigma.
    # We want to minimize this optimistic value.
    # So we maximize the negative of it: beta * sigma - mu.
    
    beta = 1.0
    scores = beta * sigma - mu
    return scores
