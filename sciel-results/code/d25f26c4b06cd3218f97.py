"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Cost-Aware Upper Confidence Bound (UCB) approach adapted for minimization.
    The score is defined as:
    Score(x) = (f_best - mu(x) + beta * sigma(x)) / cost(x)
    
    This encourages:
    1. Low predicted mean (mu) - exploitation
    2. High predicted standard deviation (sigma) - exploration
    3. Low cost - efficiency
    
    The term (f_best - mu) represents the expected improvement magnitude.
    Adding beta * sigma encourages exploration of uncertain regions.
    Dividing by cost prioritizes cheap evaluations.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Exploration-exploitation trade-off parameter
    # A value of 1.0 balances exploitation and exploration well in many BO tasks.
    # We can adjust this if needed, but 1.0 is a standard starting point for UCB.
    beta = 1.0
    
    # Calculate the cost-aware UCB score
    # We want to maximize this score.
    # (f_best - mu) is positive if mu < f_best (potential improvement).
    # beta * sigma adds an exploration bonus.
    # Dividing by cost penalizes expensive evaluations.
    
    score = (f_best - mu + beta * sigma_safe) / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
