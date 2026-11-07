"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

def _check_deadline():
    return time.time() >= _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return a safe default if we're running out of time
    if _check_deadline():
        # Return cost-based scores (prefer low cost) as a safe fallback
        return 1.0 / (cost + 1e-9)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be by definition, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # UCB with adaptive beta based on progress
    # Start with higher exploration, reduce as we get closer to budget
    # Use cost as a proxy for progress (lower cost = earlier in search)
    # Normalize cost to [0, 1] range for beta scheduling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost - cost_min) / cost_range
    
    # Beta starts high (~2.0) for exploration, decreases to ~0.5 for exploitation
    # This helps escape local optima in early iterations
    beta = 2.0 - 1.5 * cost_norm
    
    # UCB score: mu + beta * sigma (for minimization, we want low mu, high uncertainty)
    ucb = mu + beta * sigma
    
    # Combine EI and UCB: use a weighted combination
    # EI captures expected improvement, UCB captures exploration
    # Weight EI more heavily as we progress
    ei_weight = 0.7
    ucb_weight = 0.3
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) + 1e-9
    ucb_max = np.max(np.abs(ucb)) + 1e-9
    
    ei_norm = ei / ei_max
    ucb_norm = (ucb - np.min(ucb)) / (np.max(ucb) - np.min(ucb) + 1e-9)
    
    # Combined acquisition function
    combined = ei_weight * ei_norm + ucb_weight * (1 - ucb_norm)
    
    # Cost-aware penalty: use cost^1.5 for better balance than cost^2
    score = combined / (cost ** 1.5 + 1e-12)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
