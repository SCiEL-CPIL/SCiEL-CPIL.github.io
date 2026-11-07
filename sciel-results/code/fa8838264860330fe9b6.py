"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325  # 325s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement divided by cost,
    with an adaptive exploration floor to prevent premature convergence.
    
    mu, sigma, cost: shape (2000,). Higher score -> evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE - 5:
        # Return a simple fallback if we're running out of time
        return np.zeros_like(mu)

    n = mu.size

    # Ensure sigma is strictly positive for numerical stability
    sigma = np.maximum(sigma, 1e-9)

    # Standard Expected Improvement in objective units
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-awareness: divide by cost to prefer cheaper evaluations
    # cost = 1 + mean(x_scaled) where x_scaled is in [0,1]
    # So cost ranges from 1 to 2
    
    # Adaptive exploration floor:
    # When the GP is uncertain (high sigma), we want to explore even expensive regions.
    # When the GP is confident (low sigma), we should exploit cheap regions.
    # 
    # Use the median sigma as a measure of overall GP uncertainty.
    # Early on, sigma is large -> exploration floor is high -> explore more.
    # Late on, sigma is small -> exploration floor is low -> exploit more.
    sigma_median = np.median(sigma)
    
    # Exploration floor: a minimum score that scales with uncertainty
    # This prevents the acquisition from being dominated by cost alone
    # when the GP hasn't explored much yet.
    # The floor is proportional to sigma_median, so it's high when uncertain.
    exploration_floor = 0.1 * sigma_median
    
    # Cost-normalized EI
    # We want to balance EI (which favors improvement) with cost (which favors cheap points)
    # score = EI / cost encourages cheap improvements
    # But we add an exploration term to avoid getting stuck in local optima
    
    # Base score: EI divided by cost
    base_score = ei / cost
    
    # Exploration term: encourages visiting uncertain regions
    # This is proportional to sigma / cost, so uncertain AND cheap regions are favored
    # But we scale it down as the GP becomes more confident
    exploration_score = sigma / cost * (1.0 / (1.0 + sigma_median / (abs(f_best) + 1e-9)))
    
    # Combine: base_score + exploration_score, with a floor
    score = base_score + exploration_score
    
    # Apply exploration floor: ensure a minimum level of exploration
    # This prevents premature convergence by keeping a weak preference for uncertain regions
    score = np.maximum(score, exploration_floor * sigma / cost)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
