"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the Gaussian process posterior.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        Best observed value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if sigma.min() < 1e-12:
        sigma = np.maximum(sigma, 1e-12)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # EI formula: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be by construction, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring
    # We want to balance exploration (high EI) with cost efficiency
    # Using a power-law penalty: score = EI / cost^alpha
    # alpha = 2.0 gives strong preference for low-cost regions
    # This exploits the structure of Styblinski-Tang where the minimum
    # tends to be in lower-cost (more central) regions
    
    alpha = 2.0
    cost_safe = np.maximum(cost, 1e-6)
    score = ei / (cost_safe ** alpha)
    
    # Add a small exploration bonus for high uncertainty relative to cost
    # This helps escape local optima while still respecting cost
    uncertainty_bonus = sigma / (cost_safe ** 0.5)
    score = score + 0.1 * uncertainty_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
