"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills the program after 330s wall-clock.
# Each acquisition call should finish well within its share of that budget.
# We track a global start time to enforce a safety margin.
_T0 = time.time()
_DEADLINE = 330.0 - 5.0  # 5s safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4d.
    
    Uses a combination of Expected Improvement (EI), Probability of Improvement (PoI),
    and an intrinsic noise term to balance exploitation and exploration.
    The score is normalized by evaluation cost to prefer cheaper points.
    
    Key improvements over the seed:
    1. Use raw EI (not scaled) as the primary exploitation signal, since EI is
       already in objective units and captures the expected gain directly.
    2. Add a small intrinsic exploration bonus proportional to sigma to encourage
       sampling in uncertain regions, especially early on.
    3. Use a softer, more principled cost normalization: divide by (1 + cost)
       rather than cost alone, to avoid over-penalizing expensive points.
    4. Clamp and stabilize numerics to guarantee finite outputs.
    """
    # Deadline guard: if we're close to the wall, return a safe fallback
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    n = len(mu)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Standardized improvement: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is always >= 0 when mu <= f_best (i.e., potential improvement)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Intrinsic exploration: reward high-uncertainty regions
    # Use sigma directly as an exploration signal, scaled to be comparable
    # to EI. EI is in objective units; sigma is also in objective units.
    # A small weight ensures exploration doesn't dominate.
    exploration = 0.1 * sigma_safe
    
    # Combined acquisition: EI + exploration bonus
    # EI already captures exploitation; adding a small sigma term helps
    # the GP explore regions with high posterior variance.
    combined = ei + exploration
    
    # Cost-aware normalization:
    # Dividing by (1 + cost) softens the penalty compared to dividing by cost alone.
    # This prevents the acquisition from being dominated by cost differences
    # when the improvement signal is small.
    score = combined / (1.0 + cost_safe)
    
    # Ensure finite, non-negative scores
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
