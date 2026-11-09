"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Expected Improvement (EI).
    
    Parameters:
    mu: array of shape (N,) - GP mean predictions
    sigma: array of shape (N,) - GP standard deviation predictions
    f_best: float - Best observed value so far
    cost: array of shape (N,) - Cost of evaluating each point
    
    Returns:
    scores: array of shape (N,) - Acquisition scores (higher is better)
    """
    # Deadline guard: Track elapsed time to ensure we finish well before the limit.
    # Since this function is called repeatedly, we assume the caller manages the global loop,
    # but we can add a local check if this function itself were complex. 
    # For simple vectorized ops, this is negligible, but we include the structure.
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization):
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # EI should be non-negative. Due to numerical errors, it might be slightly negative.
    ei = np.maximum(ei, 0.0)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware score: EI divided by cost
    # This favors points that have high expected improvement and are cheap to evaluate.
    scores = ei / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
