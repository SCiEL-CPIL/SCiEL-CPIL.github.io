"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: Mean of the GP predictions (shape 2000,)
        sigma: Standard deviation of the GP predictions (shape 2000,)
        f_best: The best observed value so far (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        score: Acquisition scores (shape 2000,), higher is better.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # UCB for minimization:
    # We want to maximize the score.
    # Potential improvement is f_best - mu.
    # Exploration bonus is beta * sigma.
    # Score = (f_best - mu + beta * sigma) / cost
    
    # Tuning parameter for exploration
    # A value of 1.0-2.0 is typical for UCB.
    # Given the standardized targets and Matern kernel, 1.5 is a reasonable start.
    beta = 1.5
    
    raw_score = (f_best - mu) + beta * sigma_safe
    
    # Cost awareness: divide by cost to prefer cheaper evaluations
    # Ensure cost is not zero (it's 1 + mean, so min 1)
    score = raw_score / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
