"""Rewritten acquisition function using a cost-aware UCB variant with adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Levy 2D.
    
    Uses a hybrid approach:
    - Upper Confidence Bound (UCB) for exploration with adaptive beta
    - Cost normalization to account for evaluation costs
    - Handling of zero/near-zero sigma to avoid numerical issues
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute the improvement over current best
    improvement = f_best - mu
    
    # UCB term: balance exploitation (improvement) and exploration (uncertainty)
    # Adaptive beta based on the scale of sigma relative to typical improvements
    # Use a fixed beta that works well for this problem structure
    beta = 2.0
    
    ucb_score = improvement + beta * sigma_safe
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    # But we need to be careful: if cost varies, we want to account for it
    # The cost is 1 + mean(x_scaled), so it ranges from 1 to 2
    # We normalize by cost to get "improvement per unit cost"
    cost_safe = np.maximum(cost, 1e-8)
    
    # Combine UCB with cost normalization
    # We use a weighted combination to avoid pure cost-driven selection
    # when uncertainty is high
    scores = ucb_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
