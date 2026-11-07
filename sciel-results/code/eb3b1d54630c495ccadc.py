# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Compute an acquisition score for each candidate.
    Returns a (2000,) array of finite scores (higher = more preferred).
    """
    global _deadline, _start_time
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin

    # Check deadline
    if time.time() > _deadline:
        # Emergency fallback: return uniform noise or simple EI to avoid crash
        return np.ones(len(mu))

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f, 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Cost Efficiency
    # We want to favor low cost.
    # Cost is in [1, 2].
    # 1/cost is a natural measure of efficiency.
    # To make it more discriminative, we can use a power, but 1/cost is standard.
    cost_eff = 1.0 / (cost + eps)
    
    # Normalize both objectives to [0, 1] range for comparison
    # This ensures that the magnitude of EI (which depends on f_best) doesn't dominate
    # the cost term purely by scale.
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < eps:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    cost_min = np.min(cost_eff)
    cost_max = np.max(cost_eff)
    cost_range = cost_max - cost_min
    if cost_range < eps:
        cost_norm = np.ones_like(cost_eff)
    else:
        cost_norm = (cost_eff - cost_min) / cost_range

    # 3. Pareto Frontier Selection
    # A point is Pareto optimal if no other point is better in both EI and Cost Efficiency.
    # We identify the Pareto front and assign high scores to these points.
    # For points not on the front, we assign a lower score, potentially based on their
    # distance to the front or a penalty.
    
    n = len(ei_norm)
    # We will use a simple O(N^2) check if N is small, but N=2000 is 4M ops, which is fine in NumPy vectorized way?
    # Actually, O(N^2) in pure Python is slow. In NumPy, we can broadcast, but 2000x2000 matrix is 4M floats, ~32MB.
    # This is acceptable.
    
    # Create a mask: is point i dominated by point j?
    # i is dominated by j if ei_norm[j] >= ei_norm[i] AND cost_norm[j] >= cost_norm[i] AND (ei_norm[j] > ei_norm[i] OR cost_norm[j] > cost_norm[i])
    
    # To save memory and time, we can sort by one dimension and check the other.
    # Sort by ei_norm descending.
    sort_idx = np.argsort(ei_norm)[::-1]
    sorted_ei = ei_norm[sort_idx]
    sorted_cost = cost_norm[sort_idx]
    
    # In this sorted order, a point i is dominated if there exists a j < i (better or equal EI)
    # such that cost[j] >= cost[i].
    # Actually, if we sort by EI descending, then for any i, all j < i have EI[j] >= EI[i].
    # So i is dominated if max(cost[0..i-1]) >= cost[i].
    
    max_cost_so_far = -np.inf
    is_pareto = np.zeros(n, dtype=bool)
    pareto_indices = []
    
    for i in range(n):
        if sorted_cost[i] > max_cost_so_far:
            is_pareto[sort_idx[i]] = True
            pareto_indices.append(sort_idx[i])
            max_cost_so_far = sorted_cost[i]
        else:
            is_pareto[sort_idx[i]] = False

    # 4. Assign Scores
    # Points on the Pareto front get a high base score.
    # We can further rank them by a combined metric or just give them all high scores.
    # To ensure the "best" one is picked, we should differentiate among Pareto points.
    # Let's use a weighted sum for the final ranking among Pareto points,
    # but ensure non-Pareto points are significantly lower.
    
    # Base score for Pareto points: 1.0
    # Score for non-Pareto points: 0.0 (or very low)
    # Then, among Pareto points, we can use a normalized weighted sum to break ties.
    # However, the loop picks the argmax. If multiple Pareto points have score 1.0,
    # np.argmax picks the first one. This might be suboptimal.
    
    # Better: Use the weighted sum as the score, but only for Pareto points?
    # Or: Score = 1.0 for Pareto, 0.0 for others. Then add a small epsilon * weighted_sum.
    
    # Let's try: Score = is_pareto * 100.0 + (ei_norm + cost_norm) * 0.1
    # This ensures all Pareto points are vastly preferred over non-Pareto points.
    # Among Pareto points, the one with higher (ei_norm + cost_norm) is picked.
    # Since they are on the front, this effectively picks the "knee" or a balanced point.
    
    # Alternative: Just use the weighted sum directly, but heavily penalize non-Pareto.
    # Let's stick to the Pareto front logic as the primary filter.
    
    final_scores = np.zeros(n)
    if len(pareto_indices) > 0:
        # For Pareto points, use a balanced metric to select the best one.
        # Since they are non-dominated, any point on the front is valid.
        # We prefer points with higher EI and higher Cost Efficiency.
        # We already normalized them.
        # Let's use the sum of normalized scores as a tie-breaker.
        pareto_score = (ei_norm[pareto_indices] + cost_norm[pareto_indices])
        # Scale it up to be clearly above non-Pareto points
        final_scores[pareto_indices] = 10.0 + pareto_score
        
        # For non-Pareto points, give them a low score.
        # We can give them a score based on how close they are to the front,
        # or just 0.
        # Let's give them 0 to force the selection to the front.
        final_scores[~is_pareto] = 0.0
    else:
        # Fallback if no Pareto points found (should not happen)
        final_scores = ei_norm + cost_norm

    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores

# EVOLVE-BLOCK-END
