"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement and uncertainty-based exploration,
    normalized by a power of cost to prefer cheaper evaluations.
    
    Parameters:
    -----------
    mu : array, shape (2000,)
        GP mean predictions
    sigma : array, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value
    cost : array, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement (for exploration)
    pi = norm.cdf(z)
    
    # Cost-aware weighting: use cost^1.5 to more aggressively penalize expensive points
    cost_penalty = cost ** 1.5
    
    # Hybrid score: EI/cost_penalty + exploration_bonus
    # Exploration bonus encourages evaluating uncertain but cheap points
    exploration_bonus = 0.1 * sigma / cost_penalty
    
    scores = ei / cost_penalty + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
