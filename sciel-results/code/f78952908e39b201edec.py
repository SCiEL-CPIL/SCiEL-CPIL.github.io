import numpy as np
from scipy.stats import norm
import time

# Cache for timing guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4D.
    Combines Expected Improvement with an adaptive exploration term.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    global _start_time
    
    # Initialize timing guard if not set
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: return early if we're close to the 330s limit
    elapsed = time.time() - _start_time
    if elapsed > 325:  # 5 second safety margin
        # Return a simple fallback: prioritize low cost and low predicted value
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        return -np.asarray(mu, dtype=np.float64) / cost_safe
    
    # Convert inputs to numpy arrays and ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle any non-finite values in inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration parameter:
    # Use the 90th percentile of sigma as a representative uncertainty scale
    sigma_scale = np.percentile(sigma, 90)
    if not np.isfinite(sigma_scale) or sigma_scale < 1e-9:
        sigma_scale = 1.0
    
    # Base exploration weight: starts at 0.5, decreases as uncertainty shrinks
    # This encourages exploration early and exploitation later
    exploration_weight = 0.5 * (1.0 / (1.0 + sigma_scale))
    
    # Exploration term: encourages visiting uncertain regions
    exploration = exploration_weight * sigma
    
    # Combine exploitation (EI) and exploration
    combined = ei + exploration
    
    # Cost-aware acquisition: score per unit cost
    scores = combined / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
