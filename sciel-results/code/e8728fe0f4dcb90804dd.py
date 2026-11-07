"""Rewritten program with a stronger cost-aware acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 320.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    mu, sigma: shape (2000,) - GP mean and std (in objective units)
    f_best: scalar - lowest observed value
    cost: shape (2000,) - evaluation cost for each candidate
    
    Returns: shape (2000,) - scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative costs
    cost = np.maximum(cost, 1e-9)
    
    # Guard against zero/negative sigma
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute delta = f_best - mu (expected improvement baseline)
    delta = f_best - mu
    
    # Compute z = delta / sigma
    z = delta / sigma
    
    # Expected Improvement (EI)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = delta * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Expected Cost-Adjusted Improvement: EI / cost
    ei_cost = ei / cost
    
    # Uncertainty-based exploration term: sigma / cost
    # This encourages exploration in high-uncertainty, low-cost regions
    ucb_term = sigma / cost
    
    # Combine EI and exploration with a balanced weight
    # Use a time-aware or adaptive weight: early on, more exploration
    # Since we don't have iteration count, use a fixed moderate weight
    # that balances exploitation and exploration
    w_explore = 0.5
    score = ei_cost + w_explore * ucb_term
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
