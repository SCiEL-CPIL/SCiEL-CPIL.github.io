"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Expected Improvement per unit cost,
    with a small UCB exploration bonus for robustness.
    """
    # Ensure sigma is safely positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware EI
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe
    
    # Small UCB exploration bonus (beta=0.5) to avoid pure exploitation
    # This helps when the GP is uncertain, encouraging exploration
    ucb_bonus = 0.5 * sigma / cost_safe
    
    score = score + ucb_bonus
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
