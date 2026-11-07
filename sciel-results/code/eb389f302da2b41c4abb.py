"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Cost-normalized UCB with adaptive exploration.
    
    Score = (mu + kappa * sigma) / cost^gamma
    
    This aggressively explores low-cost regions with high uncertainty,
    which is critical for locating global minima in expensive optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    score : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-12)
    
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost, 1e-12)
    
    # Adaptive kappa: balance exploration and exploitation
    # Use a moderate exploration factor
    kappa = 2.0
    
    # UCB term: mu + kappa * sigma
    ucb = mu + kappa * sigma
    
    # Cost penalty with high gamma to strongly favor low-cost regions
    gamma = 3.0
    cost_penalty = cost_safe ** gamma
    
    # Final score
    score = ucb / cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
