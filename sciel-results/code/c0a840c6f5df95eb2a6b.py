# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using normalized Expected Improvement.
    
    Uses EI normalized by cost, with a careful balance between exploitation
    (high EI) and exploration (high uncertainty in promising regions).
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize the improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected improvement: E[max(f_best - f(x), 0)]
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI: primary acquisition
    # This directly optimizes improvement per unit cost
    score = ei / cost
    
    # Add a small UCB-like exploration term to avoid premature convergence
    # Beta controls exploration-exploitation tradeoff
    beta = 0.1
    ucb_term = beta * sigma_safe / cost
    
    # Final score: cost-normalized EI plus exploration bonus
    final_score = score + ucb_term
    
    # Ensure all values are finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
