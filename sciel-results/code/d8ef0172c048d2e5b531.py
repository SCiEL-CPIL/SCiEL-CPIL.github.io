"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-adaptive UCB acquisition function.
    
    Uses a hybrid approach:
    - For low-cost points: more exploration (higher beta)
    - For high-cost points: more exploitation (lower beta)
    
    The cost is normalized to [0, 1] range within the batch, and beta is
    adjusted inversely with cost to balance exploration/exploitation.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Normalize cost to [0, 1] for adaptive beta
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-12
    cost_norm = (cost - cost_min) / cost_range
    
    # Adaptive beta: lower for expensive points, higher for cheap ones
    # Base beta around 1.0-2.0, modulated by cost
    # When cost is high (cost_norm close to 1), beta is small (exploitation)
    # When cost is low (cost_norm close to 0), beta is large (exploration)
    beta_base = 1.5
    beta_min = 0.3
    beta_max = 3.0
    beta = beta_min + (beta_max - beta_min) * (1.0 - cost_norm)
    
    # Compute UCB: mu + beta * sigma
    # But we want to penalize high cost more aggressively
    # Use a cost-weighted version: divide by a cost factor
    # The cost factor should be > 1 for expensive points, < 1 for cheap points
    # This makes expensive points less attractive unless they have very high mu
    
    # Alternative: use expected improvement per unit cost with adaptive temperature
    # Let's combine: use EI-like term but with cost normalization
    
    # Expected Improvement (EI)
    z = (f_best - mu) / (sigma + 1e-12)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-normalized EI: divide by cost to get "value per cost"
    # But also add a UCB term with adaptive beta
    # Final score: combination of normalized EI and UCB
    
    # Method: Use a weighted combination
    # score = alpha * (ei / cost) + (1 - alpha) * (mu + beta * sigma) / cost
    # But this double-counts cost. Let's think differently.
    
    # Better approach: Pure cost-adaptive UCB
    # score = (mu + beta * sigma) / (cost + epsilon)
    # This naturally favors cheap points with high uncertainty or low mu
    
    epsilon = 1e-9
    ucb = mu + beta * sigma
    score = ucb / (cost + epsilon)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
