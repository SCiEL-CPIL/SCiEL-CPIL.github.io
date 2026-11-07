"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement (EI) and a penalty for high cost,
    but with a more robust formulation that handles edge cases better.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at each candidate point.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at each candidate point.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Use Lower Confidence Bound (LCB) for minimization:
    # We want to find points where the lower bound of the function value
    # is significantly better (lower) than f_best.
    # LCB(x) = mu(x) - beta * sigma(x)
    # Improvement potential = max(0, f_best - LCB(x)) = max(0, f_best - mu(x) + beta * sigma(x))
    beta = 2.0
    lcb_improvement = f_best - mu + beta * sigma_safe
    lcb_improvement = np.maximum(lcb_improvement, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    scores = lcb_improvement / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
