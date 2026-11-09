"""Cost-aware acquisition for Ackley 2D BO with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid of Expected Improvement (EI) and an exploration bonus (sigma)
    divided by cost. This combines the calibrated improvement of EI with the
    exploration benefits of UCB, which proved effective in previous runs.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Check deadline
    if _check_deadline():
        # Return a simple fallback: uniform-ish scores based on low mu
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
        # Fallback: use -mu/cost as a simple proxy
        with np.errstate(divide='ignore', invalid='ignore'):
            fallback_scores = -mu_safe / (cost_safe + 1e-9)
        fallback_scores = np.nan_to_num(fallback_scores, nan=0.0, posinf=0.0, neginf=0.0)
        return fallback_scores
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace NaN/Inf with reasonable defaults
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Hybrid: EI + Exploration Bonus
    # The previous UCB (f_best - mu + beta * sigma) scored 1.7.
    # Pure EI scored 1.0.
    # We add a scaled sigma term to EI to boost exploration in uncertain regions
    # while retaining the calibrated exploitation of EI.
    beta = 0.5
    raw_scores = ei + beta * sigma
    
    # Cost-aware: divide by cost
    scores = raw_scores / cost
    
    # Replace any remaining NaN/Inf
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
