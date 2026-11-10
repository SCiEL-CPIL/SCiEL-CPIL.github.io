"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s hard kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Quadratic-cost-penalized Expected Improvement.
    
    Uses EI with a quadratic cost penalty: score = EI / (cost * (1 + cost)).
    This strongly favors low-cost evaluations, enabling more evaluations
    within the fixed budget and better landscape sampling.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Evaluation cost for each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Time guard: if we're running low on time, return quickly
    if time.time() - _start_time > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-12)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -50, 50)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / (cost_safe * (1.0 + cost_safe))
        return np.where(np.isfinite(score), score, 0.0)
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Quadratic cost penalty: strongly favors cheap evaluations
    # cost ranges from 1 (mean of scaled coords = 0) to 2 (mean of scaled coords = 1)
    # At cost=1: denominator = 1*(1+1) = 2
    # At cost=1.5: denominator = 1.5*2.5 = 3.75
    # At cost=2: denominator = 2*3 = 6
    # This makes cheap points ~3x more attractive per unit EI than expensive ones
    cost_penalty = cost_safe * (1.0 + cost_safe)
    score = ei / cost_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
