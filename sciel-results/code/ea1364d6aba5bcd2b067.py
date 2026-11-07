"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Rank-based Pareto front selection for cost-aware Bayesian optimization.
    
    Computes expected improvement (EI) for each candidate, then identifies the
    non-dominated set in the (EI, cost) space. Candidates on the Pareto front
    receive high scores proportional to their rank within the front (sorted by EI),
    while dominated candidates receive zero score.
    """
    # 1. Compute Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    n = len(ei)
    
    # 2. Identify Non-Dominated Set (NDS) in (EI, Cost) space
    # Maximize EI, Minimize Cost.
    # Sort by EI descending, then Cost ascending.
    order = np.lexsort((cost, -ei))
    ei_sorted = ei[order]
    cost_sorted = cost[order]
    
    # We need to find points that are not dominated.
    # In the sorted list, a point is non-dominated if its cost is strictly less than 
    # the minimum cost of all points with strictly higher EI.
    # Also, for points with equal EI, only the one with the lowest cost is non-dominated 
    # (others are dominated by the lowest cost one in the same EI group).
    
    # Efficient vectorized approach:
    # 1. Find unique EI groups.
    # 2. For each group, the candidate is the first element (lowest cost).
    # 3. Check if this candidate's cost is less than the min cost of all previous groups.
    
    # Create a mask for the "representative" of each EI group (first occurrence in sorted order)
    # Since we sorted by EI desc, Cost asc, the first element of a group is the best cost for that EI.
    
    # Detect group changes
    # We treat EI values as groups. Due to floating point, we might use exact equality or a tolerance.
    # Given the nature of EI calculation, exact equality is rare but possible (e.g., sigma=0).
    # However, `np.unique` with `return_index` can help, but we need the sorted order.
    
    # Let's use a loop over groups but optimized. Or better, vectorized logic.
    # A point i in sorted order is a group leader if i==0 or ei_sorted[i] != ei_sorted[i-1].
    # Note: Using exact equality here. If there are floating point issues, we might merge/split groups incorrectly.
    # But `lexsort` groups identical values together.
    
    is_group_leader = np.zeros(n, dtype=bool)
    is_group_leader[0] = True
    # Compare consecutive EI values
    # If they are not close, it's a new group.
    # To be safe with floats, we can use a small tolerance, but exact match is standard for sorted unique logic in numpy.
    # Let's stick to exact match for speed, as EI is continuous, duplicates are unlikely unless sigma is 0.
    is_group_leader[1:] = ei_sorted[1:] != ei_sorted[:-1]
    
    # For each group leader, we check if it is non-dominated.
    # It is non-dominated if its cost < min(cost of all previous leaders).
    
    # We need the min cost of previous groups.
    # We can create an array `min_cost_prev` where `min_cost_prev[i]` is the min cost of all group leaders before i.
    
    # Let's iterate through leaders. Since number of unique EI values is likely small compared to N (2000),
    # a small loop over leaders is efficient.
    
    leader_indices = np.where(is_group_leader)[0]
    
    non_dominated_mask = np.zeros(n, dtype=bool)
    
    min_cost_so_far = np.inf
    for idx in leader_indices:
        c = cost_sorted[idx]
        if c < min_cost_so_far:
            non_dominated_mask[order[idx]] = True
            min_cost_so_far = c
    
    # 3. Assign Scores
    # Points on the front (non_dominated_mask) get scores based on their rank.
    # Rank is determined by their position in the sorted order (EI descending).
    # So the first non-dominated point in sorted order has the highest EI (among non-dominated) and gets the highest score.
    
    scores = np.zeros(n)
    
    front_indices = np.where(non_dominated_mask)[0]
    
    if len(front_indices) > 0:
        # The front_indices are already in the order of `order` which is EI desc.
        # So front_indices[0] is the highest EI front point.
        # We assign scores decreasing from 1.0 to 0.0.
        n_front = len(front_indices)
        # ranks: 0 to n_front-1
        ranks = np.arange(n_front)
        # Score: 1.0 for rank 0, 0.0 for rank n_front-1?
        # Let's use a linear decay.
        if n_front == 1:
             scores[front_indices] = 1.0
        else:
             scores[front_indices] = 1.0 - (ranks / (n_front - 1))
        
        # Optionally, we can add a small bonus to ensure they are clearly above 0.
        # But 0.0 for non-front is fine.
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
