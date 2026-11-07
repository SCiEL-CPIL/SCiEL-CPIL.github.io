import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound
    # We want to minimize f, so we maximize -mu + beta * sigma
    # Using a fixed beta for exploration
    beta = 2.0
    score = -mu + beta * sigma
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    return score
