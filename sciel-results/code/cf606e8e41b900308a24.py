import numpy as np
from scipy.stats import norm
import time

# Cost-aware acquisition function for Bayesian optimization on Levy 2D.
# We use Expected Improvement per unit cost, with a small exploration bonus
# for high-uncertainty regions, and a deadline guard for safety.

_START_TIME = None
_DEADLINE_SAFETY = 5.0  # seconds before the 330s kill limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: EI per unit cost + exploration bonus.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition values (higher = better)
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    # If we're close to the deadline, return a simple score quickly
    if elapsed > 320.0:
        # Emergency: just return EI/cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Compute z-score
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost: divide by cost to get EI per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    base_score = ei / cost_safe
    
    # Add a small exploration bonus for high-uncertainty points
    # This helps escape local optima in the Levy function
    # The bonus is scaled relative to the base score
    # Use a small coefficient to not overwhelm the exploitation term
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    exploration_bonus = 0.05 * (sigma_safe / sigma_max) / cost_safe
    
    # Combine
    score = base_score + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
