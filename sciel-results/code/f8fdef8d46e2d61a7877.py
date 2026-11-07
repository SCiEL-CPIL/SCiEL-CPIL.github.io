import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB (Upper Confidence Bound) for minimization
    # We want to maximize the score.
    # Score = -mu + beta * sigma
    # We choose beta = 1.0 as a structural change from EI.
    # EI is roughly -mu + 0.4 * sigma (max exploration term).
    # So beta=1.0 is more exploratory.
    
    beta = 1.0
    scores = -mu + beta * sigma
    return scores
