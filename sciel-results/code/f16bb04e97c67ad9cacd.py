# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement per unit cost with a small penalty for high uncertainty
    to encourage exploitation once the optimum is well-localized.
    """
    # Ensure inputs are float arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and ensure positive values
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard normal z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using scipy.stats.norm
    from scipy.stats import norm
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost
    # Add a small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    
    score = ei / cost_safe
    
    # Optional: Add a small exploration term to avoid getting stuck
    # But for minimization, high sigma means high uncertainty, which EI already handles.
    # However, if sigma is very small and mu is close to f_best, EI is small.
    # If sigma is large, EI is larger.
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
