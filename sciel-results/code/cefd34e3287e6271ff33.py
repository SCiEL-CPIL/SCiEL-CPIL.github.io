"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Uses a quadratic cost penalty (1/cost^2) to strongly prefer cheaper evaluations,
    which allows more evaluations within the fixed budget. The LCB provides a
    different exploration/exploitation balance than EI, which is better suited
    for multimodal functions like Ackley and Rastrigin.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more likely to be evaluated next)
    """
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost penalty: quadratic to strongly prefer cheap evaluations
    # cost ranges from 1.0 to 2.0, so 1/cost^2 ranges from 0.25 to 1.0
    cost_penalty = 1.0 / (cost ** 2)
    
    # Exploration parameter: balance between exploitation and exploration
    # beta controls how much we weight uncertainty
    beta = 1.5
    
    # LCB: mu - beta * sigma (lower is better for the objective)
    lcb = mu - beta * sigma_safe
    
    # Negate because higher acquisition score = more likely to evaluate
    # We want to minimize the objective, so we maximize the negative LCB
    score = -lcb * cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
