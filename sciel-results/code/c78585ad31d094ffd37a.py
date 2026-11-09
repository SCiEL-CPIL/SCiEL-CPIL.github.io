"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for Bayesian optimization.
    
    Uses a combination of:
    - UCB for exploration/exploitation balance
    - Cost-aware normalization to prefer cheaper evaluations
    - Adaptive exploration based on budget remaining
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Estimate total budget used so far (rough heuristic)
    # We don't have direct access to total_cost, but we can infer from f_best
    # and the nature of the problem. Use a simple adaptive beta.
    
    # UCB with cost-aware exploration
    # beta controls exploration: higher = more exploration
    # Scale beta inversely with cost to prefer cheaper uncertain points
    beta = 1.5  # Base exploration constant
    
    # UCB score: mu - beta * sigma (minimization problem)
    ucb = mu - beta * sigma
    
    # Cost normalization: divide by cost to get "value per unit cost"
    # But we want to balance exploration vs exploitation
    # Use a weighted combination:
    #   score = (improvement_potential) / cost + exploration_bonus
    
    # Expected improvement (for minimization)
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)  # Numerical stability
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Combine EI and UCB with cost awareness
    # Weight: give more weight to EI when we have good information,
    # more weight to UCB/exploration when uncertain
    # Adaptive weight based on sigma magnitude
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma) + 1e-9
    # Points with high sigma get more exploration weight
    exploration_weight = np.clip(sigma / sigma_max, 0.3, 1.0)
    
    # Final score: blend of EI/cost and UCB/cost
    ei_score = ei / cost
    ucb_score = ucb / cost
    
    # Blend: 70% EI, 30% UCB, with exploration weighting
    score = (0.7 * ei_score + 0.3 * ucb_score * exploration_weight)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
