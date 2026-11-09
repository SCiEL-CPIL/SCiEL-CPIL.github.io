import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Calculate Expected Improvement (EI)
    # Standard EI formula for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # z = (f_best - mu) / sigma
    
    # Handle zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values in pdf/cdf if needed, 
    # but scipy handles it well. 
    # However, for very large z, pdf is 0.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score with stronger penalty for expensive evaluations
    # Use cubic penalty to strongly favor cheaper points (low x values)
    # This is critical given the tight budget of 30 total cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / (cost_safe ** 3)
    
    # Add small exploration bonus for high uncertainty to avoid premature convergence
    # Weight by 0.1 to keep it secondary to EI
    exploration_bonus = 0.1 * sigma
    
    # Combine: cost-aware EI + exploration
    scores = scores + exploration_bonus / (cost_safe ** 2)
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
        
    return scores
