"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def _pareto_front_indices(ei, cost):
    """
    Identify indices of the Pareto front for maximizing 'ei' and minimizing 'cost'.
    A candidate i is non-dominated if there is no candidate j such that
    ei[j] >= ei[i] and cost[j] <= cost[i] with at least one strict inequality.
    """
    n = len(ei)
    is_front = np.ones(n, dtype=bool)
    
    # To speed up, we can use sorting or vectorized comparisons, but for 2000 points
    # a simple vectorized check is often fast enough in NumPy if done carefully.
    # However, O(N^2) vectorized comparisons can be memory heavy (2000x2000 = 4M floats, ~32MB).
    # This is acceptable for 2000 points.
    
    # Vectorized non-domination check:
    # For each i, check if any j dominates i.
    # j dominates i if ei[j] >= ei[i] AND cost[j] <= cost[i] AND (ei[j] > ei[i] OR cost[j] < cost[i])
    
    # We can do this by checking:
    # max_ei_for_cost_leq = max(ei[j] for j where cost[j] <= cost[i])
    # If max_ei_for_cost_leq > ei[i], then i is dominated (by some j with lower cost and higher EI).
    # Note: if max_ei == ei[i], we must check if there's a strict cost reduction.
    # Actually, if cost[j] < cost[i] and ei[j] >= ei[i], i is dominated.
    # If cost[j] == cost[i] and ei[j] > ei[i], i is dominated.
    
    # Efficient approach:
    # 1. Sort by cost ascending.
    # 2. Iterate, keeping track of max EI seen so far.
    # 3. If current EI <= max EI seen so far (from candidates with lower or equal cost), 
    #    it is dominated UNLESS it has the same cost and same EI (duplicates) or 
    #    we need to handle the "strictly better" part carefully.
    
    # Let's use the sorting approach which is O(N log N).
    order = np.argsort(cost)
    sorted_ei = ei[order]
    sorted_cost = cost[order]
    
    # We want to find indices in sorted order that form the Pareto front.
    # In sorted cost order, the Pareto front consists of points where EI is strictly increasing.
    # Why? If cost increases, we only keep it if EI also increases (otherwise the previous point with lower cost and higher/equal EI dominates it).
    
    front_mask_sorted = np.zeros(n, dtype=bool)
    
    # The first point in sorted cost order is always on the front (unless there are duplicates with lower cost, but it's the min cost).
    # Actually, if multiple points have the same min cost, the one with highest EI is on the front.
    # Let's handle groups of same cost.
    
    # Simpler: 
    # max_ei_so_far = -inf
    # for i in range(n):
    #    if sorted_ei[i] > max_ei_so_far:
    #        front_mask_sorted[i] = True
    #        max_ei_so_far = sorted_ei[i]
    #    # If sorted_ei[i] == max_ei_so_far, it's dominated by the previous point with same cost and same EI? 
    #    # No, if cost is same and EI is same, they are equivalent. 
    #    # If cost is higher and EI is same, it's dominated.
    #    # The condition "EI > max_ei_so_far" ensures we only pick points that offer a better trade-off.
    #    # Points with same cost but lower EI are skipped.
    #    # Points with higher cost but same EI are skipped (dominated by lower cost point).
    #    # Points with higher cost and higher EI are kept.
    
    max_ei_so_far = -np.inf
    for i in range(n):
        if sorted_ei[i] > max_ei_so_far:
            front_mask_sorted[i] = True
            max_ei_so_far = sorted_ei[i]
            
    # Map back to original indices
    front_mask = np.zeros(n, dtype=bool)
    front_mask[order] = front_mask_sorted
    
    return np.where(front_mask)[0]

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto front selection.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite EI
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Ensure finite cost
    cost = np.where(np.isfinite(cost), cost, np.inf)
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple EI if too slow (unlikely for 2000 points)
        return ei
    
    # Find Pareto front indices
    try:
        front_indices = _pareto_front_indices(ei, cost)
    except Exception:
        # Fallback
        return ei
    
    # Initialize scores
    scores = np.zeros_like(mu)
    
    # Assign high score to Pareto front members
    if len(front_indices) > 0:
        # We can assign a uniform high score, or rank them.
        # Uniform high score prioritizes the entire front.
        # To break ties within the front, we might prefer higher EI or lower cost, 
        # but the task says "prioritize solutions that offer the best trade-off".
        # Usually, picking any from the front is valid. 
        # However, to ensure we pick the "best" one, we can rank them by EI descending within the front.
        # But the instruction says "score candidates based on their rank within this front".
        # Let's assign scores such that front members have score 1.0 (or high), others 0.0.
        # The evaluator picks the argmax. If multiple have 1.0, it picks the first one? 
        # Usually argmax returns the first occurrence. 
        # To be more robust, let's give a slight boost to higher EI within the front?
        # Or just uniform. Uniform is the strictest interpretation of "non-dominated set".
        
        # Let's use a small epsilon decay based on rank to ensure a unique max if needed,
        # but primarily mark them as valid.
        # Actually, just setting them to 1.0 and others to 0.0 is the standard "hard" selection.
        scores[front_indices] = 1.0
        
        # Optional: If we want to prefer higher EI within the front:
        # front_ei = ei[front_indices]
        # rank = np.argsort(front_ei)[::-1]
        # scores[front_indices[rank]] = 1.0 - np.arange(len(front_indices)) * 1e-6
        # This ensures the highest EI on the front is picked first.
        
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
