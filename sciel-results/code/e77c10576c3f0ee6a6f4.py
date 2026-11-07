"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with dynamic exploration/exploitation balance.
    
    Parameters:
    mu: array of shape (N,), mean of GP
    sigma: array of shape (N,), std of GP
    f_best: float, best observed value so far
    cost: array of shape (N,), cost of evaluating each candidate
    
    Returns:
    scores: array of shape (N,), acquisition scores (higher is better)
    """
    epsilon = 1e-9
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, epsilon)
    
    # Calculate standard Expected Improvement components
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Standard EI terms
    ei_exploit = (f_best - mu) * norm.cdf(z)
    ei_explore = sigma * norm.pdf(z)
    
    # Dynamic Exploration Weight
    # Use f_best as a proxy for progress. As f_best approaches 0 (min),
    # we assume we are later in the budget and should exploit more.
    # Scale: Assume f_best > 10 is "early", f_best < 1 is "late".
    # p is progress (0=early, 1=late)
    # We want beta to decrease from 1.0 (standard) to 0.5 (exploit heavy)
    
    # Clip f_best to avoid negative progress if f_best < 0 (shouldn't happen, min is 0)
    f_best_clipped = np.clip(f_best, 0.0, 10.0)
    
    # Linear interpolation:
    # f_best=10 -> p=0 -> beta=1.0
    # f_best=0  -> p=1 -> beta=0.5
    p = 1.0 - (f_best_clipped / 10.0)
    beta = 1.0 - 0.5 * p
    
    # Adjusted EI
    ei_adj = ei_exploit + beta * ei_explore
    
    # Cost-aware score
    # Normalize by cost to favor cheaper evaluations
    scores = ei_adj / (cost + epsilon)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
