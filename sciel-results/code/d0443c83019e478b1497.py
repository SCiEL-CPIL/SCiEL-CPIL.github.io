"""
Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses Cost-aware Quantile of the Posterior (QoP) for minimization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard constants
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s limit

def _check_deadline():
    """Check if we are close to the time limit."""
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    Implements Cost-aware Quantile of the Posterior (QoP).
    We maximize the negative of the lower p-th quantile of the predictive distribution
    divided by the cost.
    
    For minimization, we want to find regions where the objective value is likely to be low.
    The p-th quantile of N(mu, sigma^2) is mu + sigma * z_p, where z_p is the p-th quantile of the standard normal.
    Using a lower quantile (e.g., p=0.1, z_p ~ -1.28) ensures we target areas that are
    "confidently low".
    
    Score = - (mu + sigma * z_p) / cost
    Since we want to maximize the score, and we want low (mu + sigma * z_p),
    maximizing -(...) is correct.
    
    z_p for p=0.1 is approximately -1.2816.
    So Score = - (mu - 1.2816 * sigma) / cost = (-mu + 1.2816 * sigma) / cost.
    This is equivalent to UCB with beta=1.2816, normalized by cost.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check deadline
    if _check_deadline():
        # Return a simple, fast score to avoid timeout
        cost_safe = np.maximum(cost, 1e-9)
        sigma_safe = np.maximum(sigma, 1e-9)
        # Fallback to simple UCB-like score
        scores = (-mu + 1.28 * sigma_safe) / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Constants for QoP
    # Using 10th percentile (z = -1.2816) for high confidence in low values
    # This balances exploitation (low mu) and exploration (high sigma)
    z_quantile = -1.2816 
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Quantile of the posterior: q = mu + sigma * z_quantile
    # We want to minimize q.
    # To maximize a score, we use -q.
    # Cost-aware: divide by cost.
    scores = (-mu - sigma_safe * z_quantile) / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
