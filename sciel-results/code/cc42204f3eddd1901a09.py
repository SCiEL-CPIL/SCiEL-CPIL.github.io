"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function: Probability of Improvement (POI) per unit cost.
    
    POI rewards points with high probability of beating f_best, which encourages
    exploration and can help escape the narrow valley of Rosenbrock more effectively
    than EI, which tends to be too conservative.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Guard against division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Probability of Improvement: P(f(x) < f_best)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Divide by cost for cost-awareness
    score = poi / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
