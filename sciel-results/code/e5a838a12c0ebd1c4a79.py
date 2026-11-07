"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track global start time for deadline safety
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 330s kill

# Constants
MIN_POSSIBLE = -0.8
MAX_POSSIBLE = 1.0  # Approx max of cosine8

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Strategy:
    Sample a single function value y ~ N(mu, sigma) for each candidate.
    Return (f_best - y) / cost as the score.
    This stochastically balances exploration and exploitation without
    requiring complex normalization or heuristic beta parameters.
    """
    # Deadline guard
    if time.time() - _START_TIME > 330.0 - _DEADLINE_MARGIN:
        return np.zeros_like(mu)

    n = len(mu)
    
    # Numerical safety
    # Ensure sigma is strictly positive for sampling
    sigma_safe = np.maximum(sigma, 1e-8)
    # Cost is in [1, 2], ensure no zeros
    cost_safe = np.maximum(cost, 1.0)
    
    # Thompson Sampling: draw one sample per candidate
    # y ~ N(mu, sigma)
    rng = np.random.default_rng()
    samples = rng.standard_normal(n)
    y = mu + sigma_safe * samples
    
    # Score: improvement over f_best, normalized by cost
    # We want to maximize this score
    improvement = f_best - y
    score = improvement / cost_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
