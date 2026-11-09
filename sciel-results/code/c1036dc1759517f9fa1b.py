"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0


def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement values."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _compute_ucb(mu, sigma, f_best, beta):
    """Compute UCB values (higher is better for minimization)."""
    return mu + beta * sigma


def _compute_pareto_scores(cost, ei, ref_cost):
    """
    Compute hypervolume contribution in 2D (cost, -EI) space.
    We minimize cost and maximize EI, so we minimize (cost, -EI).
    """
    n = len(cost)
    if n == 0:
        return np.array([])
    
    # Only consider points with positive EI (they can contribute to hypervolume)
    obj = np.column_stack([cost, -ei])
    ref_point = np.array([ref_cost, 0.0])
    
    # Sort by cost
    order = np.argsort(cost)
    sorted_cost = cost[order]
    sorted_neg_ei = -ei[order]
    
    contributions = np.zeros(n)
    
    # For each point in cost-sorted order, find its unique hypervolume contribution
    # A point's contribution is the area of the rectangle it uniquely dominates
    # that is not dominated by any other point with lower cost
    
    # Track the best (lowest) -EI seen so far from the right
    # Actually, let's think differently:
    # Sort by cost ascending. For point i, it dominates a region if no point
    # with lower cost has -EI <= sorted_neg_ei[i].
    
    # Find for each point the next point (in cost order) that has -EI <= current
    # The unique cost range is from current cost to that next point's cost
    # (or ref_cost if no such point exists)
    
    # Efficient approach: for each point, the unique contribution in the
    # second objective dimension is bounded by the minimum -EI of all
    # subsequent points (higher cost).
    
    min_neg_ei_from_right = np.minimum.accumulate(sorted_neg_ei[::-1])[::-1]
    
    for i in range(n - 1):
        # Cost width: from sorted_cost[i] to sorted_cost[i+1]
        # But only if this point is not dominated by a previous point
        # (i.e., its -EI is lower than all previous points' -EI)
        
        # Check if this point is on the Pareto front
        # A point is Pareto-optimal if no other point has both lower cost
        # and lower -EI (higher EI)
        
        # The unique hypervolume for point i:
        # cost range: [sorted_cost[i], next_cost_bound]
        # -EI range: [sorted_neg_ei[i], min(min_neg_ei_from_right[i+1:], ref_point[1])]
        
        # Actually, simpler: the contribution is the area of the rectangle
        # [sorted_cost[i], cost_upper] x [sorted_neg_ei[i], ref_point[1]]
        # minus areas covered by subsequent points.
        
        # Let's use the standard 2D hypervolume algorithm:
        # Sort by first objective. For each point, the contribution is:
        # (cost[i+1] - cost[i]) * (ref - ei[i]) - (cost[i+1] - cost[i]) * (ref - ei[i+1])
        # if ei[i+1] < ei[i], otherwise 0.
        
        # This is getting complex. Let me use a cleaner approach.
        pass
    
    # Cleaner 2D hypervolume:
    # Sort by cost ascending. The hypervolume contribution of point i is:
    # width * height where:
    #   width = cost[i+1] - cost[i] (or ref_cost - cost[i] for last)
    #   height = max(0, ref_point[1] - sorted_neg_ei[i])
    # but only if point i is not dominated.
    
    # A point is dominated if there exists j < i (lower cost) with
    # sorted_neg_ei[j] <= sorted_neg_ei[i].
    
    # Find Pareto front: points where -EI is strictly decreasing as cost increases
    pareto_mask = np.ones(n, dtype=bool)
    min_ei_so_far = np.inf
    for i in range(n):
        if sorted_neg_ei[i] >= min_ei_so_far:
            pareto_mask[i] = False
        else:
            min_ei_so_far = sorted_neg_ei[i]
    
    # Compute contributions for Pareto points
    # For each Pareto point i, its contribution is:
    # (next_pareto_cost - cost[i]) * (ref - neg_ei[i])
    # where next_pareto_cost is the cost of the next Pareto point (or ref_cost)
    
    pareto_indices = np.where(pareto_mask)[0]
    if len(pareto_indices) == 0:
        return np.zeros(n)
    
    for idx, i in enumerate(pareto_indices):
        if idx < len(pareto_indices) - 1:
            next_i = pareto_indices[idx + 1]
            cost_upper = sorted_cost[next_i]
        else:
            cost_upper = ref_cost
        
        cost_width = cost_upper - sorted_cost[i]
        ei_height = ref_point[1] - sorted_neg_ei[i]
        
        if cost_width > 0 and ei_height > 0:
            contributions[order[i]] = cost_width * ei_height
    
    return contributions


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a combination of UCB and
    Pareto-frontier hypervolume contribution.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Compute EI
    ei = _compute_ei(mu, sigma, f_best)
    
    # Check deadline early
    if time.time() - t_start > _DEADLINE * 0.3:
        # Fallback to simple EI/cost
        scores = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Compute UCB (for minimization, we want low mu + high sigma)
    # UCB for minimization: -mu + beta * sigma (higher is better)
    # Or: mu - beta * sigma (lower is better), so score = -mu + beta*sigma
    beta = 2.0
    ucb_scores = -mu + beta * sigma
    
    # Normalize UCB scores to [0, 1] range for combination
    ucb_min = np.min(ucb_scores)
    ucb_max = np.max(ucb_scores)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb_scores - ucb_min) / ucb_range
    
    # Compute Pareto hypervolume scores
    ref_cost = np.max(cost) + 0.1
    hv_scores = _compute_pareto_scores(cost, ei, ref_cost)
    
    # Normalize hypervolume scores
    hv_max = np.max(hv_scores) + 1e-9
    hv_norm = hv_scores / hv_max
    
    # Combine: weighted sum of normalized UCB and hypervolume
    # Give more weight to hypervolume as it explicitly models cost-EI trade-off
    weight_hv = 0.6
    weight_ucb = 0.4
    
    scores = weight_hv * hv_norm + weight_ucb * ucb_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores
    
    return scores


def cost_safe(cost):
    """Ensure cost values are safe for division."""
    return np.maximum(cost, 1e-6)
# EVOLVE-BLOCK-END
