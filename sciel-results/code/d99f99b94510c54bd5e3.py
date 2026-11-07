"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: balance expected improvement against evaluation cost.
    Uses EI per unit cost with a small exploration bonus for uncertain regions.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: track elapsed time to avoid timeout
    # The evaluator kills after 330s, so we add a safety margin
    t_start = time.time()
    
    # Convert inputs to arrays and ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-6)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # numerical stability
    
    # Expected improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Soft cost penalty: raise PI to power 1/cost
    # When cost is low (~1), exponent is ~1, so score ≈ PI
    # When cost is high (~2), exponent is ~0.5, so score = sqrt(PI)
    pi_cost_aware = np.power(np.clip(pi, 1e-10, 1.0 - 1e-10), 1.0 / cost)
    
    # Blend EI/cost and cost-aware PI
    # EI/cost gives magnitude of improvement per cost unit
    # pi_cost_aware gives probability of improvement adjusted for cost
    # Blend them: 0.6 * EI/cost + 0.4 * pi_cost_aware
    ei_cost_aware = ei / cost
    scores = 0.6 * ei_cost_aware + 0.4 * pi_cost_aware
    
    # Add a small exploration bonus for high uncertainty
    # This encourages exploring regions with high sigma
    exploration_bonus = sigma * 0.05
    
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - t_start
    if elapsed > 320:  # Safety margin before 330s limit
        # Return the current scores immediately
        return scores
    
    return scores
# EVOLVE-BLOCK-END
