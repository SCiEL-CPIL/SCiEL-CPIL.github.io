"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    Maximizes the score -mu + beta * sigma, which is equivalent to minimizing
    the lower confidence bound mu - beta * sigma.
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    start_time = time.time()
    deadline = start_time + 295.0

    sigma_safe = np.maximum(sigma, 1e-9)

    # LCB for minimization: we want to find the minimum.
    # The lower bound of the credible interval is mu - beta * sigma.
    # We want to select the point with the lowest lower bound.
    # So we maximize the negative of the lower bound: -mu + beta * sigma.
    
    # Choose beta. A value of 1.0 is a standard heuristic for exploration.
    # We can tune this if needed, but 1.0 is a robust default.
    beta = 1.0
    
    scores = -mu + beta * sigma_safe

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return scores

    return scores
# EVOLVE-BLOCK-END
