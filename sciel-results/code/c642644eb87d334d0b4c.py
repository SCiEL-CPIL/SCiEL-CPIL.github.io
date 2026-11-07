"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a blend of Expected Improvement (EI) and Upper Confidence Bound (UCB),
    normalized by cost, to balance exploitation, exploration, and efficiency.
    """
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound for minimization (we want low mu, high sigma)
    # UCB = mu - kappa * sigma (lower is better)
    # For maximization of score, we use -UCB = -mu + kappa * sigma
    # Let's use a kappa that balances exploration
    kappa = 1.5
    ucb_score = -mu + kappa * sigma
    
    # Blend EI and UCB
    # EI is in objective units, UCB is also in objective units
    # We normalize both by cost to get "improvement per unit cost"
    ei_per_cost = ei / cost
    ucb_per_cost = ucb_score / cost
    
    # Combine them with a weight. EI is generally more reliable, 
    # but UCB can help in early exploration.
    # Use a fixed blend for simplicity and stability.
    weight_ei = 0.7
    weight_ucb = 0.3
    
    scores = weight_ei * ei_per_cost + weight_ucb * ucb_per_cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
