"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
# Set a deadline slightly before the hard kill to allow for safe return
_DEADLINE = _start_time + 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a composite score that balances
    probability of improvement, expected reduction, and cost efficiency.
    
    Args:
        mu: Mean predictions from GP (shape 2000,)
        sigma: Std dev predictions from GP (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Scores for each candidate (higher is better)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Deadline guard
    if time.time() > _DEADLINE - 2.0:
        scores = -mu / np.where(cost > 1e-9, cost, 1.0)
        return np.where(np.isfinite(scores), scores, 0.0)

    cost_safe = np.where(cost > 1e-9, cost, 1.0)
    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute z-score for each candidate
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)

    # Probability of improvement: P(f(x) < f_best)
    p_improve = norm.cdf(z_clipped)

    # Expected improvement (standard EI for minimization)
    ei = (f_best - mu) * p_improve + sigma * norm.pdf(z_clipped)

    # Cost efficiency: expected improvement per unit cost
    ei_cost_ratio = ei / cost_safe

    # UCB term for exploration: balances exploitation (mu) with exploration (sigma)
    # Use a beta parameter that adapts based on uncertainty
    beta = 0.5
    ucb = -mu + beta * sigma

    # Normalize each component to [0, 1] range for fair combination
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    term1_norm = normalize(p_improve * ei)
    term2_norm = normalize(ei_cost_ratio)
    term3_norm = normalize(ucb)

    # Combine with weights that emphasize cost-efficiency and exploration
    scores = 0.3 * term1_norm + 0.3 * term2_norm + 0.4 * term3_norm

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Shift to ensure non-negative scores for better numerical stability
    scores = scores - np.min(scores)
    
    return scores
# EVOLVE-BLOCK-END
