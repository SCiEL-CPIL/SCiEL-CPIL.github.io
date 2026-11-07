import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound (LCB) for minimization
    # Approach: UCB family, distinct from EI.
    # We maximize the potential for finding a lower value, accounting for uncertainty.
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize LCB, so we maximize -LCB(x) = -mu(x) + beta * sigma(x)
    
    # Exploration parameter
    # Beta controls the trade-off between exploitation (low mu) and exploration (high sigma)
    beta = 2.0
    
    # Calculate the unnormalized score
    # Higher score indicates a more promising candidate (lower expected value or higher uncertainty)
    score_unnormalized = -mu + beta * sigma
    
    # Cost-aware score: higher is better
    # Divide by cost to penalize expensive evaluations (cost > 0)
    score = score_unnormalized / cost
    
    return score
