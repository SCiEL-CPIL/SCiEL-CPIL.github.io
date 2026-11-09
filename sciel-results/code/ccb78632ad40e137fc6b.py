# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with dynamic exploration budget.
    
    Uses a UCB-style score where the exploration weight is scaled by a proxy for
    remaining budget (global uncertainty) and the candidate's cost.
    Score = -mu/cost + beta * (median_sigma * sigma) / cost^2
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure safety for division
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Proxy for "Remaining Budget" / Global Exploration Potential
    # High median sigma implies early stage / high uncertainty -> more exploration
    global_uncertainty = np.median(sigma)
    
    # Base UCB weight
    # Tuned for the specific cost range [1, 2] and typical GP scales
    beta = 2.5
    
    # Exploitation term: -mu / cost
    # We want to minimize mu, so -mu is high for good candidates.
    # Dividing by cost makes it cost-efficient.
    exploit_score = -mu / cost
    
    # Exploration term: beta * global_uncertainty * sigma / cost^2
    # Scaled by global uncertainty to fade out exploration as GP converges.
    # Scaled by 1/cost^2 to heavily penalize expensive exploration.
    explore_score = beta * global_uncertainty * sigma / (cost * cost)
    
    # Total score
    scores = exploit_score + explore_score
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
