"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    Returns scores where higher values indicate more desirable candidates.
    """
    # Ensure sigma is strictly positive to avoid division issues in potential future extensions
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB exploration parameter
    # Higher beta encourages more exploration.
    # For minimization, we maximize (f_best - mu + beta * sigma) / cost
    beta = 2.5
    
    # Calculate the acquisition score
    # Exploitation term: f_best - mu (expected improvement if mu is accurate)
    # Exploration term: beta * sigma (penalty for uncertainty, added to potential improvement)
    # Cost normalization: divide by cost to prefer cheaper evaluations
    
    # Avoid division by zero (cost is >= 1, so safe, but good practice)
    safe_cost = np.maximum(cost, 1e-9)
    
    scores = (f_best - mu + beta * sigma) / safe_cost
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
