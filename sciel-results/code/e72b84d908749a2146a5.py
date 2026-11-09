"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    UCB-like hybrid acquisition: f_best - mu + beta * sigma, normalized by cost.
    This balances exploitation (low mu) with exploration (high sigma).
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB component: reward for low mean and high uncertainty
    ucb = -mu + 1.5 * sigma
    
    # Expected improvement component (for exploitation)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Hybrid: combine UCB exploration with EI exploitation
    # Weight shifts over time implicitly through the cost normalization
    # Here we use a fixed blend that favors exploration early
    score = 0.7 * ucb + 0.3 * ei
    
    # Normalize by cost to prefer cheaper evaluations
    score = score / np.maximum(cost, 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
