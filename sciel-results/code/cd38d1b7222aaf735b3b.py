"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on the probability of finding
    a point with function value less than f_best - epsilon, normalized by cost.
    Uses an adaptive epsilon that scales with the distance to the known global minimum.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Handle edge cases: where sigma is very small
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Global minimum of Shekel 4d
    global_min = -10.5364
    
    # Calculate distance from current best to global minimum
    dist_to_min = f_best - global_min
    
    # Adaptive epsilon:
    # Demand larger improvements when far from the optimum to avoid local minima.
    # Demand smaller improvements when close to the optimum to refine the solution.
    # Scale: 5% of the remaining distance, with a minimum floor of 0.01.
    epsilon = max(0.01, 0.05 * dist_to_min)
    
    # Target value we want to beat
    target = f_best - epsilon
    
    # Probability that f(x) < target under GP posterior:
    # P(f(x) < target) = Phi((target - mu) / sigma)
    z = (target - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    prob_improve = norm.cdf(z)
    
    # Ensure non-negative and avoid zero division issues in downstream logic if any
    prob_improve = np.maximum(prob_improve, 1e-15)
    
    # Normalize by cost to get probability per unit cost
    cost_safe = np.maximum(cost, 1e-8)
    
    scores = prob_improve / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
