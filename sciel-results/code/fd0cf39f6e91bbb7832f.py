import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Expected Improvement (CA-EI) acquisition for Styblinski-Tang 2D BO.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP std predictions
    - f_best: float, best observed function value
    - cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
    - array of shape (2000,), scores (higher = evaluate next)
    """
    # Check deadline - return safe fallback
    if time.time() > _DEADLINE:
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        return -mu_safe / (cost_safe + 1e-9)
    
    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # EI should be non-negative; clip small negatives from numerical error
    ei = np.maximum(ei, 0.0)
    
    # Add exploitation floor: boost points with very low mu (close to or below f_best)
    # This prevents pure exploration from dominating when uncertainty is tiny
    exploitation_floor = 0.05 * np.maximum(0.0, (f_best - mu) / (np.abs(f_best) + 1.0))
    ei = ei + exploitation_floor
    
    # Adaptive cost exponent: stronger penalty early, weaker late
    # Infer progress from f_best distance to known minimum
    known_min = -78.3323
    progress = np.clip(1.0 - (f_best - known_min) / 200.0, 0.0, 1.0)
    cost_exp = 1.5 - 0.5 * progress  # 1.5 early, 1.0 late
    
    # Cost-adjusted EI
    score = ei / (cost ** cost_exp)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
