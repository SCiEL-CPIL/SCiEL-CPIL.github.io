import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with adaptive cost penalty.
    """
    t_start = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Cost normalization: scale cost to [0, 1] range for consistent weighting
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.ones_like(cost) * 0.5
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z) where Z = (f_best - mu) / sigma
    improvement = f_best - mu
    Z = improvement / sigma
    
    # Clip Z for numerical stability
    Z = np.clip(Z, -50, 50)
    
    EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    EI = np.maximum(EI, 0.0)
    
    # Cost-awareness: penalize expensive evaluations
    # Use a moderate gamma to balance exploration and cost
    gamma = 1.0
    cost_factor = 1.0 + gamma * cost_norm
    
    # Apply cost-awareness to EI
    score = EI / cost_factor
    
    # Final normalization for stability
    score_mean = np.mean(score)
    score_std = np.std(score)
    if score_std < 1e-8:
        score_std = 1e-8
    
    score = (score - score_mean) / score_std
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - t_start > deadline:
        return score
    
    return score
