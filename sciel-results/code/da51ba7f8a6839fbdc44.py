"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of expected improvement (EI) and a penalty for
    uncertainty to balance exploration and exploitation, while accounting
    for evaluation cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure no negative or zero costs
    cost = np.maximum(cost, 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Normalize f_best relative to the range of mu for better scaling
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range < 1e-9:
        mu_range = 1.0
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    cost_aware_ei = ei / cost
    
    # Add exploration bonus based on uncertainty relative to current best
    # Use a temperature-like parameter to balance exploration
    sigma_mean = np.mean(sigma)
    if sigma_mean < 1e-9:
        sigma_mean = 1.0
    
    # Exploration bonus: higher uncertainty = higher bonus
    # Scale by how far we are from convergence
    improvement_ratio = np.maximum(f_best - mu_min, 0.0) / (improvement_ratio := max(f_best - mu_min, 1e-9))
    
    # Simple exploration-exploitation balance
    # Weight EI more heavily when we have significant improvement potential
    base_score = cost_aware_ei
    
    # Add a small exploration component to avoid getting stuck
    exploration = sigma / (sigma + 0.1 * np.std(sigma + 1e-9))
    base_score = base_score + 0.1 * exploration
    
    # Check deadline
    if time.time() > deadline:
        return np.maximum(base_score, 0.0)
    
    # Ensure all values are finite
    base_score = np.where(np.isfinite(base_score), base_score, 0.0)
    
    # Ensure non-negative
    base_score = np.maximum(base_score, 0.0)
    
    return base_score
# EVOLVE-BLOCK-END
