import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_time_limit = 330.0  # Total wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a combination of Expected Improvement (EI) and Probability of Improvement (PI)
    with adaptive exploration based on uncertainty, normalized by cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean of scaled coordinates)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the limit, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > _time_limit - 5.0:
        # Fallback: use PI as a simple, fast score
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        score = pi / np.maximum(cost, 1e-9)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Adaptive exploration: combine EI and PI with a weight based on uncertainty
    # Higher uncertainty (larger sigma) should encourage more exploration
    # We use a softmax-like weighting to balance exploitation (EI) and exploration (PI)
    
    # Normalize sigma to [0, 1] range for weighting
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Weight for exploration: increases with uncertainty
    # Use a smooth function to blend between exploitation and exploration
    exploration_weight = 0.5 + 0.5 * sigma_norm  # ranges from 0.5 to 1.0
    
    # Combine EI and PI
    # When uncertainty is high, rely more on PI (exploration)
    # When uncertainty is low, rely more on EI (exploitation)
    combined = exploration_weight * pi + (1 - exploration_weight) * ei
    
    # Cost-aware score: normalize by cost
    # Higher cost means less value per unit, so we divide by cost
    cost_safe = np.maximum(cost, 1e-9)
    score = combined / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
