"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI) + Exploration (UCB-like)
    # PI focuses on points likely to beat f_best, helping escape local minima.
    # Exploration term rewards high uncertainty, encouraging global coverage.
    # We use a relatively high exploration weight to counteract the tendency 
    # of PI to cluster around the current best if the GP is overconfident.
    
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Probability of Improvement: P(f(x) < f_best)
    # Assuming f(x) ~ N(mu, sigma^2), P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi_score = norm.cdf(z)
    
    # Exploration: sigma
    # We scale sigma to be comparable to pi_score (which is in [0,1])
    # A typical scale for sigma in this problem might be around 0.1-1.0
    # Let's use a weight that makes exploration significant but not dominant initially
    # As the search progresses, f_best decreases, making PI more selective.
    # A fixed weight of 0.5-1.0 is reasonable.
    exploration_weight = 0.8
    exploration_score = exploration_weight * sigma_safe
    
    # Combine
    scores = pi_score + exploration_score
    
    return scores
# EVOLVE-BLOCK-END
