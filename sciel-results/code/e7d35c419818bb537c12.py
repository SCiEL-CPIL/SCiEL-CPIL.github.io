"""Cost-aware acquisition for Ackley 2D BO using Pareto-frontier hypervolume contribution."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    if _check_deadline():
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
        with np.errstate(divide='ignore', invalid='ignore'):
            fallback_scores = -mu_safe / (cost_safe + 1e-9)
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)

    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Compute Expected Improvement (higher is better for minimization)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # PDF of standard normal
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    # CDF of standard normal
    cdf_z = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    
    # EI for minimization: (f_best - mu) * Phi(z) + sigma * pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # We want to maximize EI and minimize cost.
    # Let improvement = ei (higher is better)
    # Let cost = cost (lower is better)
    
    # To find Pareto front: a point is on the front if no other point has
    # both higher EI and lower cost.
    
    # Sort by EI descending, then by cost ascending for tie-breaking
    order = np.lexsort((cost, -ei))
    sorted_ei = ei[order]
    sorted_cost = cost[order]
    
    # Find Pareto front in sorted order
    # A point is dominated if there exists a previous point with cost <= current cost
    # Since we sorted by EI desc, any previous point has EI >= current EI.
    # So if previous point has cost <= current cost, current is dominated.
    
    # We can track the minimum cost seen so far.
    # If current cost is less than the min cost so far, it's on the front.
    
    # A point i is on the Pareto front if sorted_cost[i] == min_cost_prefix[i]
    # (approximately, but since we want strict improvement, we check if it's a new minimum)
    # Actually, if sorted_cost[i] < min_cost_prefix[i-1] (for i>0), it's on the front.
    # For i=0, it's always on the front.
    
    is_pareto = np.zeros(n, dtype=bool)
    is_pareto[0] = True
    min_cost_so_far = sorted_cost[0]
    for i in range(1, n):
        if sorted_cost[i] < min_cost_so_far:
            is_pareto[i] = True
            min_cost_so_far = sorted_cost[i]
    
    # Now compute hypervolume contribution for each point on the Pareto front.
    # The hypervolume contribution of a point (e_i, c_i) in the Pareto front
    # relative to the origin (0, max_cost) or similar reference.
    
    # Let's define the reference point.
    # We want to maximize EI and minimize cost.
    # The "ideal" point is (max_ei, min_cost).
    # The "nadir" point is (0, max_cost).
    # The hypervolume of a point (e, c) is the area of the rectangle from (0, max_cost) to (e, c).
    # Area = e * (max_cost - c)
    
    # However, this is not the standard hypervolume contribution which considers the union of rectangles.
    # The contribution of a point is the area it adds to the union of all rectangles.
    
    # For a sorted Pareto front (EI desc, Cost asc), the contribution of point i is:
    # (e_i - e_{i+1}) * (c_max - c_i)
    # where e_{n} = 0 and c_max is the maximum cost in the set (or a fixed upper bound).
    
    # Let's use c_max = max(cost) as the reference.
    c_max = np.max(sorted_cost)
    
    # Initialize contributions to 0
    contributions = np.zeros(n)
    
    # For each point on the Pareto front, calculate its contribution
    # We need to map back to original indices
    pareto_indices = order[is_pareto]
    pareto_ei = sorted_ei[is_pareto]
    pareto_cost = sorted_cost[is_pareto]
    
    # The contributions are calculated in the sorted Pareto front order
    # Let's say the Pareto front points are p_0, p_1, ..., p_{m-1}
    # p_0 has highest EI, lowest cost among front? No, p_0 has highest EI.
    # p_{m-1} has lowest EI, highest cost among front.
    
    # Contribution of p_i:
    # Width in EI: pareto_ei[i] - (pareto_ei[i+1] if i < m-1 else 0)
    # Height in Cost: c_max - pareto_cost[i]
    
    m = len(pareto_ei)
    for i in range(m):
        ei_next = pareto_ei[i+1] if i < m-1 else 0.0
        width = pareto_ei[i] - ei_next
        height = c_max - pareto_cost[i]
        # Ensure non-negative
        width = max(0.0, width)
        height = max(0.0, height)
        contrib = width * height
        contributions[pareto_indices[i]] = contrib
    
    # Points not on the Pareto front get 0 contribution.
    
    # Normalize or use directly?
    # The score should be higher for points with larger hypervolume contribution.
    # We can use the contribution directly.
    
    scores = contributions
    
    # Add a small epsilon to avoid ties or zero scores for non-Pareto points?
    # The task says "higher = evaluate next".
    # If a point is not on the Pareto front, it's dominated, so it should not be chosen.
    # So 0 is fine for dominated points.
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores
# EVOLVE-BLOCK-END
