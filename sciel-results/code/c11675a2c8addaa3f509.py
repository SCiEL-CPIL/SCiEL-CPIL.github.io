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
        # Emergency fallback: return inverse cost (cheapest)
        return 1.0 / np.maximum(cost, 1e-10)

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f, 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Lower Confidence Bound (LCB) based score
    # We want to minimize f. LCB = mu - beta * sigma.
    # We want low LCB values. So score = -LCB = -mu + beta * sigma.
    # Beta controls exploration. 
    # In high dimensions, exploration is hard. 
    # Let's try a moderate beta.
    beta = 1.5
    ucb_score = -mu + beta * sigma_safe
    
    # 3. Normalize both objectives to comparable scales
    # This ensures that the magnitude of EI (which depends on f_best) doesn't dominate
    # the UCB term purely by scale.
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < eps:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range < eps:
        ucb_norm = np.ones_like(ucb_score)
    else:
        ucb_norm = (ucb_score - ucb_min) / ucb_range

    # 4. Combine EI and UCB with a weighted sum
    # EI is generally the most robust.
    # UCB helps escape local minima by exploring high uncertainty regions.
    # Let's try a blend that slightly favors exploration over the pure EI seed.
    w_ei = 0.7
    w_ucb = 0.3
    
    combined = w_ei * ei_norm + w_ucb * ucb_norm
    
    # 5. Compute Cost Efficiency
    # We want to favor low cost.
    # Cost is in [1, 2].
    # 1/cost is a natural measure of efficiency.
    cost_eff = 1.0 / (cost + eps)
    
    cost_min = np.min(cost_eff)
    cost_max = np.max(cost_eff)
    cost_range = cost_max - cost_min
    if cost_range < eps:
        cost_norm = np.ones_like(cost_eff)
    else:
        cost_norm = (cost_eff - cost_min) / cost_range

    # 6. Pareto Frontier Selection
    # A point is Pareto optimal if no other point is better in both Combined Score and Cost Efficiency.
    # We identify the Pareto front and assign high scores to these points.
    # For points not on the front, we assign a lower score.
    
    n = len(ei_norm)
    
    # Sort by combined score descending.
    sort_idx = np.argsort(combined)[::-1]
    sorted_combined = combined[sort_idx]
    sorted_cost = cost_norm[sort_idx]
    
    # In this sorted order, a point i is dominated if there exists a j < i (better or equal Combined)
    # such that cost[j] >= cost[i].
    # Actually, if we sort by Combined descending, then for any i, all j < i have Combined[j] >= Combined[i].
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

    # 7. Assign Scores
    # Points on the Pareto front get a high base score.
    # We can further rank them by a combined metric or just give them all high scores.
    # To ensure the "best" one is picked, we should differentiate among Pareto points.
    # Let's use the combined score for the final ranking among Pareto points,
    # but ensure non-Pareto points are significantly lower.
    
    final_scores = np.zeros(n)
    if len(pareto_indices) > 0:
        # For Pareto points, use the combined score to select the best one.
        # Since they are non-dominated, any point on the front is valid.
        # We prefer points with higher Combined Score and higher Cost Efficiency.
        # We already normalized them.
        # Let's use the sum of normalized scores as a tie-breaker.
        pareto_score = (combined[pareto_indices] + cost_norm[pareto_indices])
        # Scale it up to be clearly above non-Pareto points
        final_scores[pareto_indices] = 10.0 + pareto_score
        
        # For non-Pareto points, give them a low score.
        # We can give them a score based on how close they are to the front,
        # or just 0.
        # Let's give them 0 to force the selection to the front.
        final_scores[~is_pareto] = 0.0
    else:
        # Fallback if no Pareto points found (should not happen)
        final_scores = combined + cost_norm

    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores

# EVOLVE-BLOCK-END
