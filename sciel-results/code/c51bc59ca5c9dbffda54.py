"""Cost-aware acquisition for griewank-2d BO using Probability of Improvement (PoI)
with cost-aware scaling. PoI focuses on the likelihood of improvement rather than
expected magnitude, which is more robust for cost-aware selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # seconds, with safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """Cost-aware Probability of Improvement acquisition function.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute improvement: how much better could we do?
    improvement = f_best - mu
    
    # Probability of improvement: P(f(x) < f_best)
    # Given GP posterior N(mu, sigma), this is:
    # P(f < f_best) = Phi((f_best - mu) / sigma) = Phi(improvement / sigma)
    z = improvement / sigma
    poi = norm.cdf(z)
    
    # Cost-aware scaling: penalize expensive evaluations
    # Cost ranges from 1.0 (cheapest, x near 0) to 2.0 (most expensive, x near 1)
    # We want to prefer cheaper points when improvement is similar
    # Use a soft penalty: score = PoI / cost^alpha
    # alpha controls the strength of cost awareness
    
    # Adaptive alpha: stronger cost penalty when we have lots of budget left
    # (early on, we can afford to explore; later, be more cost-efficient)
    # But we don't know remaining budget exactly, so use a moderate fixed value
    alpha = 0.5
    
    # Cost scaling: divide by cost^alpha to penalize expensive points
    cost_scaled = np.power(np.maximum(cost, 0.1), alpha)
    
    # Combine: higher PoI and lower cost = higher score
    scores = poi / cost_scaled
    
    # Add small exploration bonus for very uncertain points
    # This helps when PoI is similar but one point has much higher uncertainty
    # The bonus is small so it doesn't override the main PoI signal
    sigma_max = np.max(sigma)
    if sigma_max > 1e-6:
        exploration_bonus = 0.05 * (sigma / sigma_max)
        scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check: if deadline is approaching, return quickly
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Return a simple fallback: just use PoI without cost scaling
        scores = np.where(np.isfinite(poi), poi, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
