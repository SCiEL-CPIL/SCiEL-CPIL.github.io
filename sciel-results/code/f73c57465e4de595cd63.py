import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    UCB balances exploration and exploitation by maximizing the upper bound
    of the negative function value (equivalent to minimizing the lower bound).
    Score = -mu + beta * sigma
    
    This is a structural change from the previous hybrid EI approach,
    replacing the ad-hoc bonus with a principled uncertainty weighting.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and non-zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # UCB parameter (exploration rate)
    # A value of 2.0 is a standard default for 3D optimization problems
    # where the objective scale is O(1).
    beta = 2.0
    
    # Calculate UCB score for minimization
    # We want to maximize the score, which corresponds to finding
    # points with low predicted mean and high uncertainty.
    scores = -mu + beta * sigma_safe
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
