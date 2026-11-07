"""Rewritten program using Cost-Adjusted UCB (Upper Confidence Bound)."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Upper Confidence Bound (UCB).
    
    For minimization, we use an optimistic bias: we prefer points with low predicted
    mean (exploitation) and high uncertainty (exploration).
    
    Score = (-mu + beta * sigma) / cost
    
    This balances exploitation and exploration while penalizing expensive evaluations.
    """
    # Exploration-exploitation trade-off parameter
    # A value of 1.5 is a common heuristic for balanced exploration
    beta = 1.5
    
    # Calculate UCB score for minimization
    # We want to maximize this score: low mu, high sigma
    ucb_score = -mu + beta * sigma
    
    # Cost-aware weighting
    # Divide by cost to penalize expensive points
    score = ucb_score / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
