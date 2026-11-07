"""Cost-aware acquisition for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# --- module-level state -------------------------------------------------
_t0 = time.monotonic()
_HARD_DEADLINE = _t0 + 295.0     # hard wall-clock cap for the whole run
_EPS = 1e-12


def acquisition(mu, sigma, f_best, cost):
    # --- deadline guard: return a safe, finite, deterministic answer fast ---
    if time.monotonic() > _HARD_DEADLINE:
        cost_safe = np.maximum(np.asarray(cost, dtype=float), _EPS)
        return np.ones(len(mu)) / cost_safe          # favour cheap points

    t_start = time.monotonic()
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    n = len(mu)

    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=_EPS)

    # Ensure sigma and cost are positive
    sigma = np.maximum(sigma, _EPS)
    cost = np.maximum(cost, _EPS)

    # 1) Compute Expected Improvement (EI) for minimization
    #    Improvement = f_best - mu
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)  # numerical safety
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative

    # 2) Structural change: Rank-based Pareto front selection
    #    We compute a non-dominated set based on (EI, -cost) where:
    #    - Higher EI is better (we want high expected improvement)
    #    - Lower cost is better (we want low evaluation cost)
    #    
    #    A candidate i is dominated by j if:
    #    ei[j] >= ei[i] AND cost[j] <= cost[i] AND at least one strict inequality
    #
    #    Non-dominated candidates get high scores, dominated ones get low scores.
    #    Within the non-dominated set, we rank by a weighted combination to break ties.

    # Efficient Pareto front detection using numpy vectorization
    # For each point, check if any other point dominates it
    # A point i is dominated if exists j such that ei[j] >= ei[i] AND cost[j] <= cost[i] 
    # AND (ei[j] > ei[i] OR cost[j] < cost[i])

    # We use a sorting-based approach for efficiency:
    # Sort by EI descending, then for each point, check if the minimum cost so far 
    # (from points with higher or equal EI) is less than or equal to current cost.
    
    # Sort indices by EI descending, break ties by cost ascending
    ei_sorted_idx = np.argsort(-ei, kind='mergesort')
    ei_sorted = ei[ei_sorted_idx]
    cost_sorted = cost[ei_sorted_idx]
    
    # For each position in the sorted array, compute the running minimum cost
    # A point at position k is dominated if there exists a point at position < k
    # (i.e., with higher or equal EI) that has cost <= current cost
    # Since we sorted by EI descending, all previous points have EI >= current EI
    # So we just need to check if min(cost[0:k]) <= cost[k]
    
    # Running minimum of cost from the start
    min_cost_so_far = np.minimum.accumulate(cost_sorted)
    
    # A point is non-dominated if its cost is STRICTLY LESS than the minimum cost 
    # of all points with strictly higher EI. But since we have ties in EI, we need care.
    # Simpler: a point is dominated if there exists another point with ei >= ei_i and cost <= cost_i
    # and at least one strict. In the sorted order (EI desc), for point at index k,
    # all points j < k have ei[j] >= ei[k]. If min(cost[0:k]) < cost[k], then dominated.
    # If min(cost[0:k]) == cost[k], we need to check if any has strictly higher EI or strictly lower cost.
    # This is tricky. Let's use a simpler heuristic that is fast and effective:
    
    # Alternative: Use a weighted score for ranking within the front, but first identify the front.
    # For efficiency with 2000 points, O(n^2) is 4M ops, which is fine in numpy.
    
    # Let's do the O(n^2) check but vectorized per row to keep it fast.
    # Actually, the sorting approach is O(n log n). Let's refine it.
    
    # A point i is non-dominated if no point j has ei[j] >= ei[i] and cost[j] <= cost[i] with at least one strict.
    # In the sorted-by-EI-descending array, for point at index k:
    # All j < k have ei[j] >= ei[k].
    # If min(cost[0:k]) < cost[k], then dominated (there's a point with >= EI and strictly lower cost).
    # If min(cost[0:k]) == cost[k], then there might be a point with equal EI and equal cost (duplicate) or 
    # a point with higher EI and equal cost. In either case, if there's a point with strictly higher EI and equal cost,
    # it dominates. If only equal EI and equal cost, it's a tie (not dominated in strict sense, but we can handle).
    # For simplicity and speed, we'll consider a point dominated if min(cost[0:k]) <= cost[k] and 
    # (there exists a point with strictly higher EI or strictly lower cost among 0..k-1).
    
    # Let's just compute: is there any j < k such that cost[j] < cost[k] OR (cost[j] == cost[k] AND ei[j] > ei[k])?
    # Since ei is sorted descending, for j < k, ei[j] >= ei[k].
    # So the condition is: min(cost[0:k]) < cost[k] OR (min(cost[0:k]) == cost[k] AND ei[0:k] has any value > ei[k])
    
    # This is getting complex. Let's use a simpler, robust O(n^2) approach since n=2000 is small.
    
    # Vectorized O(n^2) dominance check:
    # For each i, check if any j dominates i.
    # ej >= ei AND cj <= ci AND (ej > ei OR cj < ci)
    
    # We can do this with broadcasting but it uses 2000x2000 = 4M floats = 32MB, which is fine.
    ei_diff = ei[:, None] - ei[None, :]  # ei[i] - ei[j]
    cost_diff = cost[:, None] - cost[None, :]  # cost[i] - cost[j]
    
    # j dominates i if ei[j] >= ei[i] AND cost[j] <= cost[i] AND (ei[j] > ei[i] OR cost[j] < cost[i])
    # ei[j] >= ei[i]  <=>  ei[i] - ei[j] <= 0  <=>  ei_diff[i,j] <= 0
    # cost[j] <= cost[i] <=> cost[i] - cost[j] >= 0 <=> cost_diff[i,j] >= 0
    # ei[j] > ei[i] <=> ei_diff[i,j] < 0
    # cost[j] < cost[i] <=> cost_diff[i,j] > 0
    
    dom_mask = (ei_diff <= 0) & (cost_diff >= 0) & ((ei_diff < 0) | (cost_diff > 0))
    
    # A point i is dominated if any j (j != i) dominates it.
    # Note: dom_mask[i,i] is False because ei_diff[i,i]=0 and cost_diff[i,i]=0, so (ei_diff<0)|(cost_diff>0) is False.
    dominated = dom_mask.any(axis=1)
    non_dominated = ~dominated
    
    # 3) Assign scores based on Pareto front membership
    #    Non-dominated points get high scores, dominated points get low scores.
    #    Within non-dominated points, rank by a combined score to differentiate.
    
    # For non-dominated points, use a weighted combination of EI and inverse cost for ranking
    # Normalize EI and cost to [0, 1] for the ranking
    ei_max = np.max(ei) + _EPS
    ei_min = np.min(ei)
    ei_norm = (ei - ei_min) / (ei_max - ei_min + _EPS)
    
    cost_max = np.max(cost) + _EPS
    cost_min = np.min(cost)
    cost_norm = (cost - cost_min) / (cost_max - cost_min + _EPS)
    
    # Combined score for ranking non-dominated points: higher EI and lower cost is better
    # Use 0.7 * EI_norm + 0.3 * (1 - cost_norm)
    rank_score = 0.7 * ei_norm + 0.3 * (1.0 - cost_norm)
    
    # Assign scores:
    # Non-dominated: score in [0.6, 1.0] based on rank_score
    # Dominated: score in [0.0, 0.3] based on rank_score (lower)
    
    scores = np.zeros(n, dtype=np.float64)
    
    # For non-dominated points
    if np.any(non_dominated):
        nd_rank_scores = rank_score[non_dominated]
        nd_max = np.max(nd_rank_scores) + _EPS
        nd_min = np.min(nd_rank_scores)
        nd_norm = (nd_rank_scores - nd_min) / (nd_max - nd_min + _EPS)
        # Map to [0.6, 1.0]
        scores[non_dominated] = 0.6 + 0.4 * nd_norm
    
    # For dominated points
    if np.any(dominated):
        d_rank_scores = rank_score[dominated]
        d_max = np.max(d_rank_scores) + _EPS
        d_min = np.min(d_rank_scores)
        d_norm = (d_rank_scores - d_min) / (d_max - d_min + _EPS)
        # Map to [0.0, 0.3]
        scores[dominated] = 0.0 + 0.3 * d_norm
    
    # 4) Add a small uncertainty bonus to encourage exploration
    #    This helps when the Pareto front is large and we need to differentiate further
    max_sigma = np.max(sigma) + _EPS
    uncertainty_bonus = 0.1 * (sigma / max_sigma)
    
    scores += uncertainty_bonus
    
    # 5) finite / bounded output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.clip(scores, -1e6, 1e6)

    return scores
# EVOLVE-BLOCK-END
