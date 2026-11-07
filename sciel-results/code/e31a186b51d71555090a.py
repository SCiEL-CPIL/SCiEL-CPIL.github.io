"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard: 330s wall-clock limit, keep 20s safety margin.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # 300s per evaluation, but we have 330s total for the whole loop
# Actually the contract says "Each evaluation must finish within 300 s" and the evaluator
# kills after 330s. Since acquisition() is called once per iteration, and there are many
# iterations, we need to be careful. But the function itself should be fast.
# Let's use a per-call deadline with a small budget.

def _time_remaining():
    return _DEADLINE - time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Expected Improvement per unit cost,
    with a structural change to use a robust variant that handles
    uncertainty better and includes a small exploration bonus.
    
    mu, sigma: shape (2000,), GP mean and std in objective units
    f_best: scalar, best observed value
    cost: shape (2000,), evaluation cost for each candidate
    
    Returns: shape (2000,), finite scores (higher = evaluate next)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety: return zeros if anything is off
    if n == 0:
        return np.array([])
    
    # Clamp sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standard EI computation
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with norm.cdf/pdf for extreme values
    z = np.clip(z, -50.0, 50.0)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost
    # cost = 1 + mean(x) where x in [0,1]^6, so cost in [1, 2]
    cost_safe = np.maximum(cost, 1e-10)
    scores = ei / cost_safe
    
    # Structural change: add a small UCB-like exploration term weighted by
    # the cost-awareness, to encourage exploration in uncertain regions
    # that aren't already well-covered. This uses a temperature-like
    # parameter that balances exploitation vs exploration.
    # The exploration term is sigma / cost, which rewards high-uncertainty
    # low-cost points.
    
    # Adaptive exploration weight: start moderate, could be tuned
    # Use a fixed small weight to avoid over-exploration
    exploration_weight = 0.1
    exploration = exploration_weight * sigma_safe / cost_safe
    
    scores = scores + exploration
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
