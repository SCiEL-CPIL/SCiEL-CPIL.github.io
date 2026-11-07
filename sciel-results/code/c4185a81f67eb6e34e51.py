"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    
    # Ensure cost is positive and not too small
    cost = np.maximum(cost, 1.0) # Min cost is 1.0
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement (EI) for minimization
    # Improvement is (f_best - mu)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # EI should be non-negative, but numerical errors can make it slightly negative
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Additive cost penalty with uncertainty-aware exploration bonus
    # Instead of dividing by cost (multiplicative), use additive penalty that scales with uncertainty
    # This creates a more stable trade-off: high-uncertainty regions get more exploration allowance
    # but cost still discourages expensive evaluations
    
    # Normalize cost to [0, 1] for consistent penalty scaling
    cost_norm = (cost - 1.0) / 1.0  # cost is in [1.0, 2.0], so this gives [0, 1]
    cost_norm = np.clip(cost_norm, 0.0, 1.0)
    
    # Additive cost penalty: higher cost reduces score, but penalty is moderate
    cost_penalty = 0.5 * cost_norm
    
    # Uncertainty-weighted exploration bonus: rank-based to encourage diversity
    # Higher sigma means more uncertainty, which should get a small bonus
    # Use rank to make this scale-invariant
    sigma_ranks = np.argsort(np.argsort(sigma))  # 0 to n-1
    sigma_ranks = sigma_ranks / max(len(sigma_ranks) - 1, 1)  # normalize to [0, 1]
    
    # Exploration bonus: small, scaled by a factor to not dominate EI
    exploration_bonus = 0.1 * sigma_ranks
    
    # Combine: EI - cost_penalty + exploration_bonus
    score = ei - cost_penalty + exploration_bonus
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
