"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling-style acquisition for Bayesian optimization of Ackley 2D.
    
    Scores candidates using a dynamic exploration strategy:
    - Early on (large f_best): aggressively explore high-variance regions
    - Late on (f_best near 0): sharply focus on the global minimum
    
    The exploration weight is scaled by the current best value to balance
    exploration and exploitation adaptively.
    """
    sigma = np.maximum(sigma, 1e-9)
    
    # Dynamic exploration weight based on current best value
    # When f_best is large (early), we want more exploration
    # When f_best is small (late), we want more exploitation
    # Scale factor: as f_best approaches 0, exploration weight decreases
    
    # Use a smooth transition based on f_best
    # f_best ranges from ~0 (optimal) to larger values (early)
    # We want exploration_weight to be high when f_best is large, low when small
    
    # Normalize f_best to get a sense of progress
    # Ackley 2D minimum is 0, so f_best in [0, ~10] typically
    # Use exponential decay for the exploration weight
    exploration_weight = np.exp(-2.0 * f_best)
    
    # Quantile-normalized standard deviation term
    # This helps identify truly high-variance regions relative to the distribution
    sigma_sorted = np.sort(sigma)
    sigma_median = np.median(sigma)
    
    # Create a dynamic sigma term that emphasizes regions with above-median uncertainty
    # but also scales with the overall uncertainty level
    sigma_term = sigma / (sigma_median + 1e-9)
    
    # Apply a non-linear transformation to make the exploration more aggressive
    # in high-variance regions
    sigma_term = np.power(sigma_term, 1.5)
    
    # Base score: negative mean (exploitation)
    base_score = -mu
    
    # Exploration bonus: weighted by dynamic exploration weight
    exploration_bonus = exploration_weight * sigma_term
    
    # Combined score
    score = base_score + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
