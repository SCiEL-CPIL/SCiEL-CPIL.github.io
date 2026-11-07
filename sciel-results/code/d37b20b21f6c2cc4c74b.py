"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: total wall-clock budget is 330s, keep safety margin
_DEADLINE_START = time.time()
_TIME_LIMIT = 320.0  # seconds, leave 10s margin before 330s kill


def _time_remaining():
    return _TIME_LIMIT - (time.time() - _DEADLINE_START)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    mu: (2000,) predicted mean in objective units
    sigma: (2000,) predicted std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) evaluation cost for each candidate
    
    Returns: (2000,) scores, higher = prefer to evaluate next
    """
    # Check time budget
    if _time_remaining() < 5.0:
        # Return a simple fallback: just -mu / cost to be safe
        return -mu / (cost + 1e-9)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamically adjust kappa based on how much of the budget is consumed
    # cost is the per-candidate cost; we need to estimate total budget consumed
    # The loop runs while total cost < 30. We don't know total cost directly,
    # but we can use the fact that we're being called repeatedly.
    # A simpler approach: use the median cost as a proxy for typical cost,
    # and assume we've used a fraction of the budget.
    # Actually, we don't have access to the total budget consumed.
    # Let's use a heuristic: as f_best decreases, we should exploit more.
    # Or we can use the ratio of sigma to f_best as a proxy for "how much
    # uncertainty remains relative to the current best."
    
    # Use a dynamic kappa that decreases as f_best approaches 0
    # When f_best is large (early), explore more (higher kappa)
    # When f_best is small (late), exploit more (lower kappa)
    # f_best for Griewank 2D starts around 0.5-1.0 and goes to 0
    
    # Normalize f_best: assume it ranges from ~1.0 to 0.0
    f_best_clamped = np.clip(f_best, 0.0, 1.0)
    # kappa starts high (exploration) and decreases to low (exploitation)
    # Use a smooth transition
    kappa = 2.0 * (1.0 - f_best_clamped) + 0.5  # ranges from 0.5 to 2.5
    
    # LCB score: maximize (mu - kappa * sigma) / cost
    # Since we want to minimize, lower mu is better, so we use -(mu - kappa*sigma)
    # Wait: LCB for minimization is mu - kappa * sigma. We want to maximize this
    # (i.e., find the most optimistic lower bound). But lower values of the objective
    # are better, so we want candidates where mu - kappa*sigma is as low as possible.
    # So the score should be -(mu - kappa * sigma) / cost = (kappa * sigma - mu) / cost
    
    lcb_score = (kappa * sigma_safe - mu) / (cost + 1e-9)
    
    # Also compute EI for comparison
    delta = f_best - mu
    z = delta / sigma_safe
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei_score = ei / (cost + 1e-9)
    
    # Use max(EI, LCB) to avoid additive interference
    scores = np.maximum(ei_score, lcb_score)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
