import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Bayesian optimization."""
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Expected improvement
    delta = f_best - mu
    z = delta / sigma_safe
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to zero (can't have negative improvement)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI per unit cost, with a small exploration floor
    # The floor prevents the algorithm from completely ignoring expensive regions
    # that might contain the optimum
    base_score = ei / cost_safe
    
    # Add a small exploration bonus proportional to uncertainty
    # This helps escape local exploitation and explore the domain
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    
    score = base_score + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
