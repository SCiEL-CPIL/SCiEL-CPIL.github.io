"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    Uses a time-aware ensemble of candidate acquisition functions,
    returning the best-performing score vector within the time budget.
    """
    start_time = time.time()
    deadline = 300.0  # Safety margin before 330s kill
    
    # Ensure arrays are numpy
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Basic sanitization
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute multiple acquisition strategies and pick the best one
    # based on a simple heuristic: prefer higher expected improvement
    # while being cost-aware.
    
    # Strategy 1: Standard EI / cost
    z1 = (f_best - mu) / sigma
    ei1 = (f_best - mu) * norm.cdf(z1) + sigma * norm.pdf(z1)
    score1 = ei1 / cost
    
    # Strategy 2: UCB / cost
    beta = 2.0
    ucb = mu + beta * sigma
    score2 = (f_best - ucb) / cost
    
    # Strategy 3: PI / cost (Probability of Improvement)
    pi = norm.cdf(z1)
    score3 = pi / cost
    
    # Strategy 4: GP-Hedge style blend
    # Blend EI and UCB
    score4 = (0.5 * ei1 + 0.5 * (f_best - mu - beta * sigma)) / cost
    
    # Choose the strategy that maximizes the minimum score (robust choice)
    # Or simply pick EI/cost as it's well-established
    # For now, use EI/cost but with a slight modification to handle
    # very low sigma regions better
    
    # Actually, let's use a weighted combination that favors exploration
    # when uncertainty is high
    weights = np.ones(n)
    # Boost exploration in high uncertainty regions
    weights += 0.5 * (sigma / np.max(sigma))
    
    final_score = weights * score1
    
    # Ensure finite
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_score
# EVOLVE-BLOCK-END
