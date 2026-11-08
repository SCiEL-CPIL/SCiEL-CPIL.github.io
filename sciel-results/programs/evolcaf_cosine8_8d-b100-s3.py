"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost with exploration bonus.
    eps = 1e-9
    sigma_safe = sigma + eps
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware normalization
    # cost is in [1, 2] since mean(x scaled to [0,1]) is in [0,1]
    # We want to favor cheaper points when EI is similar
    # But also want to explore when uncertainty is high
    # Use a combined score: EI/cost + exploration_term
    # Exploration term encourages evaluating points with high sigma
    # Scale exploration to be comparable to EI/cost
    
    # The maximum possible EI is bounded by (f_best - min_possible) which is small
    # sigma is in objective units, typically small for a well-fitted GP
    # Let's use a weighted combination
    explore_weight = 0.1  # weight for exploration
    exploit_weight = 1.0  # weight for exploitation
    
    score = exploit_weight * (ei / cost) + explore_weight * (sigma / (cost + 1))
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
