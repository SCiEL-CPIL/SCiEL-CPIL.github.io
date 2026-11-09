"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Cost-Aware Expected Improvement with Exploration Bonus.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI)
    # Minimization problem: EI = E[max(f_best - f(x), 0)]
    xi = 0.01
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Exploration Bonus
    # Add a small multiple of sigma to encourage exploration in uncertain regions
    # This is distinct from UCB as it adds to EI rather than replacing it
    alpha = 0.05
    exploration_bonus = alpha * sigma_safe
    
    # Combined Score before cost
    raw_score = ei + exploration_bonus
    
    # Cost Awareness
    # Divide by cost to prefer cheaper evaluations
    # Cost is > 1
    score = raw_score / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
