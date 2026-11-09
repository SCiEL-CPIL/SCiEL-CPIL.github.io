"""Rewritten program with Cost-Aware EI and Exploration Bonus."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with an additive exploration bonus, normalized by cost.
    
    Parameters:
    mu: array of shape (N,), GP mean predictions
    sigma: array of shape (N,), GP standard deviation predictions
    f_best: float, best observed value so far
    cost: array of shape (N,), evaluation costs
    
    Returns:
    scores: array of shape (N,), acquisition scores (higher is better)
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the standard normal CDF and phi is the PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add an exploration bonus proportional to uncertainty
    # This encourages exploration in areas with high variance,
    # even if the mean prediction is not significantly better than f_best.
    exploration_beta = 0.5
    exploration_bonus = exploration_beta * sigma_safe
    
    # Combine EI and exploration bonus
    total_improvement = ei + exploration_bonus
    
    # Normalize by cost to account for evaluation expenses
    # Cost is in range [1, 2]
    scores = total_improvement / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
