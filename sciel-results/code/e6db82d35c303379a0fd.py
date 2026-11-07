import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function with adaptive exploration.
    """
    t_start = time.time()
    deadline = t_start + 290.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety
    sigma = np.clip(sigma, 1e-9, None)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 1e-6), cost, 1.0)

    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration: add a small uncertainty bonus
    # This helps explore when EI is low (exploitation phase)
    # The bonus is proportional to sigma but scaled down
    # to avoid excessive exploration
    exploration_bonus = 0.1 * sigma
    
    # Combine EI and exploration
    score = ei + exploration_bonus
    
    # Cost-aware normalization: prefer cheaper points
    # Use 1/cost for strong preference for cheap evaluations
    cost_weight = 1.0 / cost
    
    score = score * cost_weight
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
