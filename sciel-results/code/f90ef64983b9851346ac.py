# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    This approach uses a different family than Expected Improvement (EI).
    UCB balances exploitation (low predicted mean) and exploration (high uncertainty).
    For minimization, we maximize:
        Score = (-mu + beta * sigma) / cost
    
    This encourages selecting points that are either predicted to be good 
    (low mu) or have high potential to be good due to uncertainty (high sigma),
    while penalizing expensive evaluations.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Exploration parameter
    # Beta controls the trade-off between exploitation and exploration.
    # A higher beta leads to more exploration.
    beta = 2.0
    
    # Calculate UCB score
    # -mu favors points with low predicted function values.
    # beta * sigma favors points with high uncertainty.
    ucb_raw = -mu + beta * sigma_safe
    
    # Normalize by cost
    scores = ucb_raw / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
