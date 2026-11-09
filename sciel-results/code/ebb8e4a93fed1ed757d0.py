"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0


def _find_nondominated(costs, eis, n_top=50):
    """
    Find the non-dominated front in (cost, -EI) space.
    We want to minimize cost and maximize EI, so non-dominated means
    no other point has both lower cost AND higher EI.
    
    Returns indices of the non-dominated set (top n_top by hypervolume contribution).
    """
    n = len(costs)
    if n == 0:
        return np.array([], dtype=int)
    
    # Sort by cost ascending, then by EI descending for ties
    order = np.lexsort(((-eis), costs))
    sorted_costs = costs[order]
    sorted_eis = eis[order]
    
    # Find non-dominated points: a point is dominated if there exists another
    # with cost <= this cost and EI >= this EI (with at least one strict)
    # Since sorted by cost asc, we just need to track max EI seen so far
    nondominated_mask = np.zeros(n, dtype=bool)
    max_ei_so_far = -np.inf
    for i in range(n):
        if sorted_eis[i] > max_ei_so_far + 1e-15:
            nondominated_mask[order[i]] = True
            max_ei_so_far = sorted_eis[i]
    
    nd_indices = np.where(nondominated_mask)[0]
    
    if len(nd_indices) <= n_top:
        return nd_indices
    
    # If too many, select top by hypervolume contribution
    # Approximate: rank by EI / cost ratio
    ratios = eis[nd_indices] / np.maximum(costs[nd_indices], 1e-6)
    top_local = np.argsort(-ratios)[:n_top]
    return nd_indices[top_local]


def _compute_hypervolume_scores(costs, eis, nd_indices):
    """
    Compute a smooth score for all points based on their hypervolume contribution
    relative to the non-dominated front. Points closer to or on the front get higher scores.
    
    Uses a soft approximation: for each point, compute the "gap" to the Pareto front
    in a normalized space, then score = 1 / (1 + gap^2).
    """
    n = len(costs)
    if len(nd_indices) == 0:
        return np.zeros(n)
    
    nd_costs = costs[nd_indices]
    nd_eis = eis[nd_indices]
    
    # Normalize cost and EI to [0, 1] range for consistent scaling
    cost_range = np.ptp(nd_costs)
    if cost_range < 1e-10:
        cost_range = 1.0
    ei_range = np.ptp(nd_eis)
    if ei_range < 1e-10:
        ei_range = 1.0
    
    cost_min_nd = np.min(nd_costs)
    ei_max_nd = np.max(nd_eis)
    
    # For each point, find the "best" non-dominated point in terms of trade-off
    # Use a smooth min over the front: for each candidate, compute the minimum
    # "distance" to any non-dominated point in normalized (cost, -EI) space
    
    # Vectorized approach: compute for each candidate the minimum over all ND points
    # of a distance metric. Use a soft-min via log-sum-exp for smoothness.
    
    # Reshape for broadcasting: (n, 1) vs (1, n_nd)
    c_diff = costs[:, None] - nd_costs[None, :]  # (n, n_nd)
    e_diff = eis[:, None] - nd_eis[None, :]       # (n, n_nd)
    
    # Normalize differences
    c_diff_norm = c_diff / cost_range
    e_diff_norm = e_diff / ei_range
    
    # Distance: we want points that are NOT dominated. A point is better if
    # it has lower cost (c_diff < 0 means candidate has lower cost) and higher EI
    # (e_diff > 0 means candidate has higher EI).
    # Define a "domination gap": positive if dominated, negative if it could be on front
    # gap = max(0, c_diff_norm) - max(0, e_diff_norm) ... this isn't quite right
    
    # Better: for each candidate, compute the minimum over ND points of:
    #   max(0, c_diff_norm[i,j]) + max(0, -e_diff_norm[i,j])
    # This is zero if the candidate dominates or equals that ND point.
    
    pos_c = np.maximum(c_diff_norm, 0.0)   # extra cost vs ND point
    neg_e = np.maximum(-e_diff_norm, 0.0)  # less EI vs ND point
    
    gap_matrix = pos_c + neg_e  # (n, n_nd), zero if candidate dominates ND point j
    
    # Soft-min over the ND dimension: score based on minimum gap
    # Use log-sum-exp for smooth min
    # min_j gap_matrix[i,j] ≈ -logsumexp(-gap_matrix[i,j])
    # But gaps can be large, so clip
    gap_clipped = np.clip(gap_matrix, 0.0, 10.0)
    
    # Soft min: lower gap is better
    # score_i = 1 / (1 + softmin_j(gap_matrix[i,j]))
    # softmin via: -logsumexp(-alpha * gap) / alpha for temperature alpha
    alpha = 5.0
    neg_alpha_gap = -alpha * gap_clipped
    # logsumexp: max + log(sum(exp(x - max)))
    row_max = np.max(neg_alpha_gap, axis=1)
    logsumexp_vals = row_max + np.log(np.sum(np.exp(neg_alpha_gap - row_max[:, None]), axis=1))
    soft_min_gap = -logsumexp_vals / alpha
    
    # Score: higher when soft_min_gap is lower
    scores = 1.0 / (1.0 + soft_min_gap ** 2)
    
    # Also blend with the raw EI/cost ratio for points far from the front
    raw_ratio = eis / np.maximum(costs, 1e-6)
    max_ratio = np.max(raw_ratio)
    if max_ratio > 1e-10:
        raw_ratio_norm = raw_ratio / max_ratio
        scores = 0.7 * scores + 0.3 * raw_ratio_norm
    
    return scores


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier hypervolume scoring.
    
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
    
    # Ensure sigma has a small floor to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Check deadline before expensive Pareto computation
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE * 0.5:
        # Fall back to simple cost-aware EI
        cost_penalized = np.maximum(cost, 1e-6) ** 1.5
        scores = ei / cost_penalized
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Pareto-frontier-based scoring
    nd_indices = _find_nondominated(cost, ei, n_top=50)
    scores = _compute_hypervolume_scores(cost, ei, nd_indices)
    
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
