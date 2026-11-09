"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware EI blended with cost-aware UCB for exploration
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB component for exploration
    # Use a beta that balances exploitation and exploration
    beta = 1.0
    ucb = -mu + beta * sigma
    
    # Blend: weight EI and UCB, both cost-aware
    # Use a fixed blend ratio that worked well in measurements
    alpha = 0.5
    score = (alpha * ei + (1 - alpha) * ucb) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
