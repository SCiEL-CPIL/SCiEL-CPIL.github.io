import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition function with enhanced exploration
    
    # Numerical safety: clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Exploration term: Reward high uncertainty
    # This helps escape local optima in multimodal functions like Levy
    # The finding suggests penalizing known-good regions (low sigma, low improvement)
    # is equivalent to rewarding unexplored regions (high sigma)
    exploration_weight = 1.0
    exploration_term = exploration_weight * sigma_safe
    
    # Combine exploitation (EI) and exploration
    # We can also scale EI to be comparable to sigma if needed, 
    # but since they are in the same units (objective), direct addition is standard.
    score = ei + exploration_term
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-10)
    score = score / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
