"""Rewritten program using a different acquisition approach: Cost-Aware Upper Confidence Bound (UCB)."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: scalar - best (lowest) observed function value so far
    - cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB for minimization: we want to minimize f(x).
    # The "upper confidence bound" for the minimum is typically:
    # UCB_min = mu - beta * sigma
    # We maximize this quantity to find points likely to be low.
    # However, we must also account for cost.
    
    # A common cost-aware UCB formulation is to maximize (f_best - (mu - beta*sigma)) / cost
    # or simply (f_best - mu + beta*sigma) / cost.
    # Let's use the standard exploration-exploitation trade-off:
    # Score = (f_best - mu) + beta * sigma
    # This rewards points with low mean (exploitation) and high uncertainty (exploration).
    # We divide by cost to prefer cheaper points.
    
    # Tuning parameter beta. 
    # A value around 1.0-2.0 is standard. 
    # Given the standardized targets and small noise, sigma represents uncertainty in the objective scale.
    # Let's use beta = 1.0 as a starting point, which balances exploitation and exploration.
    beta = 1.0
    
    # Calculate the raw UCB score for minimization
    # We want to select x that minimizes f(x).
    # The term (f_best - mu) is the expected improvement in mean.
    # The term beta * sigma is the exploration bonus.
    # Combining them: (f_best - mu) + beta * sigma
    # Wait, standard UCB for maximization is mu + beta*sigma.
    # For minimization, we want to maximize the probability of finding a low value.
    # This is equivalent to maximizing -f(x).
    # So UCB for -f(x) is -mu + beta*sigma.
    # If we define the acquisition function to maximize, we can use:
    # Score = -(mu - beta * sigma) = -mu + beta * sigma.
    # To make it relative to f_best (which helps scaling), we can add f_best:
    # Score = f_best - mu + beta * sigma
    
    raw_score = f_best - mu + beta * sigma_safe
    
    # Cost-aware score: divide by cost
    scores = raw_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
