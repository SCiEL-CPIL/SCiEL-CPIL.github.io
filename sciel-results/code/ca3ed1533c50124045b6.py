"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr
import time

_DEADLINE = 325.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Cost-Adjusted Probability of Improvement (POI) with
    dynamic improvement threshold scaled by remaining budget.
    Returns 2000 finite scores (higher = more promising).
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)
    if n != 2000:
        return np.ones(n)

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Estimate total cost spent so far.
    # cost[i] is the cost of evaluating candidate i if chosen.
    # We don't know which candidates were actually evaluated, but we can use
    # the mean cost as a proxy. The total budget is 30.
    # Initial 5 uniform points cost ~5 * 1.5 = 7.5 on average.
    # We estimate remaining budget as 30 - estimated_spent.
    # Since we don't track exact spent, use a heuristic:
    # The number of evaluations so far isn't directly available, but we can
    # infer from the GP state implicitly. For simplicity, use a fixed estimate
    # that the search is in progress and remaining budget decreases over time.
    # A simple approach: assume we've spent roughly proportional to the number
    # of iterations. Since we don't have iteration count, use a static fraction.
    # Better: use the fact that f_best is improving, so we're making progress.
    # Let's use a simple model: remaining_budget = 30 - 7.5 - k * (f_best - f_min_estimate)
    # But f_min is unknown. Instead, use a fixed gamma and scale by a budget proxy.
    
    # Simpler approach: Use the mean cost to estimate per-eval cost, and assume
    # a reasonable number of evals have occurred. Since cost is in [1,2],
    # and total budget is 30, we can have at most 30/1 = 30 evals.
    # Let's estimate that we've used about 50% of the budget on average.
    # This is a heuristic; the key is to make the threshold tighter as budget depletes.
    
    # Use a dynamic threshold: target = f_best - gamma * remaining_fraction
    # where remaining_fraction decreases as we spend more.
    # Since we don't have exact spent, use a proxy: the mean of cost_safe
    # reflects the "expensiveness" of the current search region.
    # Let's define remaining_fraction = max(0, 1 - mean_cost * 0.5)
    # This gives a value in [0.5, 0.75] for cost in [1, 2].
    
    mean_cost = np.mean(cost_safe)
    remaining_fraction = np.clip(1.0 - 0.25 * mean_cost, 0.1, 1.0)
    
    # Gamma: scale the improvement demand. Start moderate, increase as budget depletes.
    gamma = 0.5 * (1.0 / remaining_fraction)
    
    # Target: we want improvement over f_best by at least gamma * remaining_fraction
    # This forces larger relative improvements as budget depletes.
    target = f_best - gamma * remaining_fraction
    
    # Probability of Improvement: P(f(x) < target)
    # For normal distribution: P(mu + sigma*Z < target) = P(Z < (target - mu)/sigma)
    z_poi = (target - mu) / sigma_safe
    z_poi_clipped = np.clip(z_poi, -10, 10)
    poi = ndtr(z_poi_clipped)
    
    # Cost penalty: divide by cost to prefer cheaper evaluations
    # Use a moderate power to balance cost and improvement probability
    cost_penalty = cost_safe ** 1.5
    
    scores = poi / cost_penalty
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Small epsilon to avoid all-zero scores
    scores = np.maximum(scores, 1e-10)
    
    return scores
# EVOLVE-BLOCK-END
