import numpy as np
from scipy.stats import norm
import time

# Deadline guard: kill after 330s wall-clock, return with safety margin
_DEADLINE = 330.0
_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME, _DEADLINE
    
    # Check deadline - if we're running out of time, return a safe fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Return uniform scores (safe, finite)
        return np.ones_like(mu, dtype=np.float64)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Small epsilon for numerical stability
    eps = 1e-12
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # Handle sigma ~ 0 carefully
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical errors can cause tiny negative values)
    ei = np.maximum(ei, 0.0)
    
    # Exploration term (UCB-like)
    # Use a moderate kappa for exploration
    kappa = 1.0
    exploration = kappa * sigma
    
    # Combine EI and exploration
    # We want to maximize this score
    score = ei + exploration
    
    # Cost-aware: divide by cost
    # cost is shape (2000,), score is shape (2000,)
    # Avoid division by zero, though cost >= 1
    score = score / (cost + eps)
    
    # Ensure finite scores
    # Replace any non-finite values with 0 or a very small number
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
