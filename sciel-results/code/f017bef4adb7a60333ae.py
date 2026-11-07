"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety deadline
_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin below 330s


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive LCB with cost-efficiency weighting.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Adaptive LCB-based acquisition with cost-efficiency
    # Key improvements:
    # 1. Use a blend of EI and LCB to balance probabilistic and heuristic signals
    # 2. Adaptive beta that scales with the uncertainty landscape
    # 3. Cost penalty that is adaptive based on the improvement potential
    
    # Compute LCB improvement potential
    # LCB for minimization: L(x) = mu(x) - beta * sigma(x)
    # Improvement potential from LCB: (f_best - mu) + beta * sigma
    
    # Adaptive beta: scale based on the ratio of max to mean sigma
    # When uncertainty is concentrated (high max/mean ratio), explore more
    # When uncertainty is uniform, use moderate exploration
    max_sigma = np.max(sigma_safe)
    mean_sigma = np.mean(sigma_safe)
    if mean_sigma < 1e-9:
        mean_sigma = 1e-9
    
    # Uncertainty concentration ratio
    sigma_ratio = max_sigma / mean_sigma
    
    # Beta adapts: higher when uncertainty is concentrated, lower when uniform
    # Base beta of 1.0, scaled by concentration (capped to avoid extremes)
    beta = 1.0 * np.clip(sigma_ratio, 1.0, 3.0)
    
    # LCB improvement potential
    lcb_improvement = (f_best - mu) + beta * sigma_safe
    
    # Blend EI and LCB: use a weighted combination
    # EI is probabilistic, LCB is heuristic. Blend gives robustness.
    # Weight: 0.6 * EI + 0.4 * LCB_improvement
    # This maintains the probabilistic foundation while adding heuristic exploration
    
    blended_improvement = 0.6 * ei + 0.4 * lcb_improvement
    
    # Adaptive cost penalty:
    # Instead of simple division, use a penalty that depends on the improvement potential.
    # High-improvement points can afford higher cost; low-improvement points need to be cheap.
    # Use: score = improvement / (cost + lambda * (1 - normalized_improvement))
    # where normalized_improvement is in [0, 1]
    
    # Normalize improvement to [0, 1] for adaptive penalty
    max_improvement = np.max(blended_improvement)
    if max_improvement < 1e-9:
        max_improvement = 1e-9
    normalized_improvement = np.clip(blended_improvement / max_improvement, 0.0, 1.0)
    
    # Adaptive cost penalty: lambda decreases as improvement increases
    # lambda = 0.5 * (1 - normalized_improvement)
    # This means: high-improvement points get less cost penalty
    lambda_cost = 0.5 * (1.0 - normalized_improvement)
    
    # Effective cost: original cost + adaptive penalty
    effective_cost = cost_safe + lambda_cost * cost_safe
    
    # Final score
    scores = blended_improvement / effective_cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
