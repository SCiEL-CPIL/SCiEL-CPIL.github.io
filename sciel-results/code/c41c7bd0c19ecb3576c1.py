"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB)
    with adaptive exploration parameter.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - lowest observed objective value
    cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero or negative sigma (though sigma should be positive)
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive exploration parameter
    # Use the gap to the known minimum (0) to scale exploration.
    # If f_best is large, we are far from the optimum, so explore more.
    # If f_best is close to 0, exploit more.
    # Using sqrt helps dampen the exploration as we get closer.
    gap = np.maximum(f_best, 1e-6)
    beta = np.sqrt(gap)
    
    # UCB score for minimization:
    # We want to find points with low mu and high sigma.
    # Score = -mu + beta * sigma
    # This score is maximized for good candidates.
    ucb_score = -mu + beta * sigma
    
    # Cost-aware normalization
    # Divide by cost to prefer cheaper points if scores are similar.
    # cost is >= 1.
    scores = ucb_score / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
