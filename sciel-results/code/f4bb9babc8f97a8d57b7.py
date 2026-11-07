"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Maximizes the optimistic improvement per unit cost:
    Score = (f_best - mu + beta * sigma) / cost
    
    Parameters:
    mu: array of predicted means (shape 2000,)
    sigma: array of predicted standard deviations (shape 2000,)
    f_best: scalar, best observed value so far
    cost: array of evaluation costs (shape 2000,)
    
    Returns:
    array of scores (shape 2000,), higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive to avoid division errors
    cost_safe = np.maximum(cost, 1e-9)
    
    # Beta controls the exploration/exploitation trade-off.
    # 2.0 corresponds to a 95% confidence interval width.
    beta = 2.0
    
    # Calculate LCB score
    # f_best - mu is the expected improvement (positive if mu < f_best)
    # beta * sigma is the exploration bonus
    score = (f_best - mu + beta * sigma) / cost_safe
    
    # Replace non-finite values with -inf to ensure they are never selected
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
