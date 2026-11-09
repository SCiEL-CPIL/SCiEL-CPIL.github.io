"""Rewritten program using Lower Confidence Bound (LCB) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound acquisition for minimization.
    
    Higher score = more promising to evaluate next.
    LCB = mu - beta * sigma
    We return -LCB so that higher values are preferred (since we want to minimize).
    Actually, let's think about this more carefully.
    
    For minimization:
    - We want to find points with low predicted mean (mu)
    - We also want to explore areas with high uncertainty (sigma)
    - LCB for minimization: mu - beta * sigma (lower LCB is better)
    - To convert to a "score" where higher is better: -LCB = -mu + beta * sigma
    
    But we need to be careful about scale. Let's use a simple LCB:
    score = -mu + beta * sigma
    
    The constant beta controls the exploration/exploitation trade-off.
    A larger beta means more exploration.
    """
    beta = 2.0
    # For minimization, LCB = mu - beta * sigma
    # We want to maximize the "desirability", so we return -LCB = -mu + beta * sigma
    # But we should also consider f_best to make it more adaptive
    
    # Simple LCB approach:
    scores = -mu + beta * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
