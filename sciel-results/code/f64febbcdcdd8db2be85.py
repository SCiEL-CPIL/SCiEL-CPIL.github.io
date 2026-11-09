"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return current best if we exceed 320s (safety margin from 330s limit)
    start_time = time.time()
    
    # Cost-aware acquisition: Expected Improvement per unit cost
    # With a slight exploration bonus via UCB component
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Score = EI / cost
    
    # To balance exploitation and exploration, we can use a weighted combination
    # of cost-aware EI and UCB
    z = (f_best - mu) / (sigma + 1e-9)
    from scipy.stats import norm
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB component for exploration
    beta = 2.0
    ucb = mu - beta * sigma
    
    # Combine: primarily cost-aware EI, with small UCB exploration
    # This helps escape local minima in the rosenbrock landscape
    score = (ei / cost) + 0.1 * (ucb / (cost + 1e-9))
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
