# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    This approach draws a sample from the GP posterior and selects the candidate
    that minimizes the sampled value divided by its cost. This treats the objective
    as a stochastic reward-to-go ratio, balancing exploration (via the sigma term)
    and exploitation (via the mu term) while accounting for evaluation costs.
    """
    # Ensure numerical stability
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Draw a sample from the standard normal distribution
    z = np.random.standard_normal(len(mu))
    
    # Thompson Sampling: sample from the GP posterior
    # f_sample = mu - sigma * z (since we want to minimize, we subtract)
    f_sample = mu - sigma_safe * z
    
    # Ensure cost is positive to avoid division by zero or negative costs
    cost_safe = np.maximum(cost, eps)
    
    # Cost-aware score: -f_sample / cost
    # We negate because we want higher scores for better (lower) values
    scores = -f_sample / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
