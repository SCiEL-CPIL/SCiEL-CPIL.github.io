# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) minus a cost penalty, plus a GP uncertainty bonus.
    Score = EI - penalty * cost + bonus * sigma / scale
    """
    # Deadline guard
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero and negative sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -30, 30)
    
    # Compute EI (Expected Improvement for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost penalty: penalize expensive evaluations
    penalty = 0.1
    cost_penalty = penalty * cost_safe
    
    # Add a small exploration bonus based on uncertainty
    # This helps explore high-uncertainty regions even when EI is low
    # The bonus is scaled by sigma relative to the current best
    scale = max(abs(f_best), 1e-6)
    exploration_bonus = 0.15 * sigma_safe / scale
    
    # Combined score: higher is better
    scores = ei - cost_penalty + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
