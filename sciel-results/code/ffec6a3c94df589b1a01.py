import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    UCB balances exploitation (low mu) and exploration (high sigma).
    For minimization, we want to minimize (mu - beta * sigma).
    The score returned should be higher for better candidates, so we return
    the negative of the UCB value: -mu + beta * sigma.
    """
    # Beta controls the exploration/exploitation trade-off.
    # A value around 2.0 is a common heuristic for Bayesian Optimization.
    beta = 2.0
    
    # Calculate UCB scores
    # Lower (mu - beta * sigma) is better.
    # We return higher scores for better candidates.
    scores = -mu + beta * sigma
    
    return scores
