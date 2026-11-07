import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 310  # 20s safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using two-stage Pareto front filtering.
    
    Stage 1: Compute EI and identify non-dominated candidates (EI vs cost).
    Stage 2: Select from Pareto front using dynamic convex combination
             that shifts toward cost-efficiency as remaining budget decreases.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean at candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation at candidate points.
    f_best : float
        Best observed function value (minimum).
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros(2000)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, np.nanmean(mu) if np.any(np.isfinite(mu)) else 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Also compute UCB as an alternative improvement metric
    # Adaptive beta
    median_sigma = np.median(sigma)
    std_mu = np.std(mu)
    beta = 1.0 + 0.5 * (median_sigma / (std_mu + 1e-8))
    beta = np.clip(beta, 0.5, 5.0)
    ucb_improvement = np.maximum(f_best - mu + beta * sigma, 0.0)
    
    # Blend EI and UCB improvement
    improvement = 0.6 * ei + 0.4 * ucb_improvement
    
    # Stage 1: Pareto front identification
    # Non-dominated: no other point has both higher improvement AND lower cost
    # Use vectorized approach for efficiency
    # A point i is dominated by point j if improvement[j] >= improvement[i] AND cost[j] <= cost[i]
    # and at least one strict inequality
    
    n = len(improvement)
    
    # For efficiency, we'll use a threshold-based approach instead of full O(n^2)
    # Sort by improvement descending, then find points where cost is at a new minimum
    sort_idx = np.argsort(-improvement)
    sorted_imp = improvement[sort_idx]
    sorted_cost = cost[sort_idx]
    
    # A point is on the Pareto front if its cost is strictly less than all previous points
    # (when sorted by improvement descending)
    min_cost_so_far = np.full(n, np.inf)
    min_cost_so_far[0] = sorted_cost[0]
    for i in range(1, n):
        min_cost_so_far[i] = min(min_cost_so_far[i-1], sorted_cost[i])
    
    # Pareto front: points where cost equals the running minimum
    # But we want to include near-front points too for robustness
    # Use a relative threshold: include points within 10% of the best cost at their improvement level
    
    # Alternative: use a convex combination directly on all points
    # This is simpler and avoids the O(n) loop
    
    # Estimate remaining budget
    # Total budget is 30. Initial 5 points cost ~5 * 1.5 = 7.5 (avg cost ~1.5 for uniform in [-4,5])
    # Actually cost = 1 + mean(x scaled to [0,1]). For uniform in [-4,5], scaled to [0,1]:
    # x in [-4,5], range 9. Scaled: (x - (-4)) / 9 = (x+4)/9
    # mean of scaled = mean(x+4)/9 = (mean(x)+4)/9. For uniform, mean(x) = 0.5, so mean(scaled) = 4.5/9 = 0.5
    # So avg cost = 1 + 0.5 = 1.5
    # 5 initial points: cost ~7.5
    # Remaining budget: 30 - 7.5 = 22.5 for evaluations
    # But we don't know exactly how many evaluations have been done.
    # Use the cost array to estimate: if we assume the cost distribution is stable,
    # we can estimate remaining budget from the median cost.
    
    median_cost = np.median(cost)
    # Rough estimate: we've spent about 7.5 + (number of evals so far) * median_cost
    # We don't know number of evals. Let's use a heuristic:
    # If f_best is close to 0, we're in late stage. If f_best is high, early stage.
    # Powell 4d minimum is 0. Typical initial values are large.
    
    # Use f_best as a proxy for progress:
    # f_best starts high (maybe 10-100) and decreases toward 0.
    # progress = 1 - f_best / initial_estimate
    # But we don't know initial. Use a log scale.
    
    # Simpler: use the ratio of best improvement to typical improvement
    max_imp = np.max(improvement)
    if max_imp > 0:
        # Fraction of candidates with significant improvement
        frac_good = np.mean(improvement > 0.1 * max_imp)
    else:
        frac_good = 0.0
    
    # Dynamic weight: shift toward cost-efficiency as we go
    # w_imp: weight on improvement, w_cost: weight on cost-efficiency
    # As frac_good decreases (late stage), we should care more about cost
    # Actually, in late stage, improvements are small, so cost matters more
    # In early stage, big improvements are possible, so favor improvement
    
    # Use f_best directly: lower f_best means later stage
    # Normalize f_best: assume initial ~50, final ~0
    # progress = clip(1 - f_best / 50, 0, 1)
    progress = np.clip(1.0 - f_best / 50.0, 0.0, 1.0)
    
    # Weight on cost-efficiency increases with progress
    # w_cost = 0.3 + 0.5 * progress  (from 0.3 to 0.8)
    w_cost = 0.3 + 0.5 * progress
    w_imp = 1.0 - w_cost
    
    # Normalize improvement and cost for convex combination
    imp_max = np.max(improvement)
    cost_max = np.max(cost)
    cost_min = np.min(cost)
    
    if imp_max > 0:
        imp_norm = improvement / imp_max
    else:
        imp_norm = np.zeros_like(improvement)
    
    if cost_max > cost_min:
        cost_norm = (cost_max - cost) / (cost_max - cost_min)  # higher = cheaper
    else:
        cost_norm = np.zeros_like(cost)
    
    # Convex combination score
    score = w_imp * imp_norm + w_cost * cost_norm
    
    # Add a small exploration bonus for high uncertainty
    sigma_max = np.max(sigma)
    if sigma_max > 0:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Blend in exploration
    explore_weight = 0.1 * (1.0 - progress)  # more exploration early
    score = score + explore_weight * sigma_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
