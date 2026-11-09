"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

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
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware score: EI divided by cost
    # This favors points that are both expected to improve the objective and cheap to evaluate
    scores = ei / cost_safe
    
    # Handle any non-finite values (should not happen with safe divisions, but just in case)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
