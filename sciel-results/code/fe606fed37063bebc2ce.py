"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track iteration count for dynamic exploration
_call_count = 0
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    global _call_count
    
    # Deadline guard: if we are running out of time, return a safe fallback
    # The task says 300s limit per evaluation. 
    # Since this function is called 2000 times per iteration? No, "evaluator scores 2000 new uniform candidates with acquisition".
    # So this function is called ONCE per iteration with arrays of length 2000.
    # Wait, "Each function runs on five seeded instances... While total cost is below 30...".
    # The loop runs while total cost < 30.
    # Each iteration:
    # 1. Generate 2000 candidates.
    # 2. Call acquisition(mu, sigma, f_best, cost) -> returns 2000 scores.
    # 3. Pick max score.
    # 4. Evaluate.
    # So acquisition is called once per BO iteration.
    # The 300s limit is for the whole evaluation? "Each evaluation must finish within 300 s."
    # This likely means the entire function call `acquisition` must be fast.
    # 2000 points is small. Pareto front calculation is O(N log N) or O(N^2) in worst case.
    # N=2000, O(N^2) is 4M ops. Very fast in numpy/python if vectorized or efficient.
    
    _call_count += 1
    
    # 1. Compute Expected Improvement (EI)
    # mu and sigma are in objective units.
    # f_best is the lowest observed value.
    # We want to minimize f.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-6)
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # 2. Identify Pareto Front
    # Objective 1: Maximize EI
    # Objective 2: Minimize Cost
    
    # A point i is dominated by point j if:
    # ei[j] >= ei[i] AND cost[j] <= cost[i] AND (ei[j] > ei[i] OR cost[j] < cost[i])
    
    # Efficient way to find non-dominated points in 2D:
    # Sort by one objective (e.g., Cost ascending).
    # Then scan from best cost to worst cost, keeping track of the max EI seen so far.
    # If current EI > max_EI_seen_so_far, it is non-dominated.
    
    # Sort indices by cost
    cost_sorted_idx = np.argsort(cost)
    costs_sorted = cost[cost_sorted_idx]
    eis_sorted = ei[cost_sorted_idx]
    
    # Initialize mask for non-dominated points
    is_pareto = np.zeros(len(cost), dtype=bool)
    
    # Scan
    max_ei_seen = -np.inf
    for i in range(len(costs_sorted)):
        if eis_sorted[i] > max_ei_seen:
            is_pareto[cost_sorted_idx[i]] = True
            max_ei_seen = eis_sorted[i]
        else:
            break # Since costs are sorted ascending, if we see a lower EI, all subsequent points have higher cost and lower/equal EI, so they are dominated by the current best EI point?
            # Wait.
            # If costs are sorted ascending: c[0] <= c[1] <= ...
            # If at index i, ei[i] <= max_ei_seen (which came from some j < i, so c[j] <= c[i]), then point i is dominated by point j.
            # However, could a point k > i have higher EI? Yes.
            # But point k has cost c[k] >= c[i].
            # If point i is dominated by j, it's not on the front.
            # We just need to identify points that are NOT dominated.
            # The logic "if ei[i] > max_ei_seen" correctly identifies the skyline.
            # We cannot break early because a later point with higher cost might have much higher EI, making it non-dominated relative to previous points, but it will be compared against the max_ei_seen which includes all previous points.
            # Actually, we should NOT break. We must scan all.
            # Correction: We do NOT break. We just don't mark it as Pareto.
            pass # Remove break logic from thought process, implement correctly below.

    # Re-implement scan without break
    max_ei_seen = -np.inf
    for i in range(len(costs_sorted)):
        if eis_sorted[i] > max_ei_seen:
            is_pareto[cost_sorted_idx[i]] = True
            max_ei_seen = eis_sorted[i]
            
    # 3. Assign Scores
    # The directive says: "assign scores proportional to their rank within this front"
    # "rank within this front" usually means the position in the sorted non-dominated set.
    # How to define rank? 
    # Typically, we might want to favor points that are "better" in some sense.
    # Since it's a front, all are non-dominated.
    # A common approach: Assign higher score to points with higher EI (or lower cost).
    # Let's assign scores based on their index in the Pareto set sorted by EI descending?
    # Or simply: Points on the front get a high base score, others get 0.
    # To distinguish among Pareto points, we can use their normalized EI or Cost.
    
    # Let's refine "proportional to their rank".
    # If we have K Pareto points, we can assign scores 1/K, 2/K, ... K/K?
    # Which point gets which rank?
    # Let's sort the Pareto points by Cost (ascending).
    # The cheapest point on the front might be considered "rank 1" (best for budget)?
    # Or the highest EI point?
    # Let's assign higher scores to points with higher EI, as EI is the primary objective.
    # Among Pareto points, sort by EI descending.
    # Rank 1 (highest EI) gets highest score.
    
    pareto_indices = np.where(is_pareto)[0]
    if len(pareto_indices) == 0:
        # Fallback if something went wrong (should not happen)
        return np.zeros_like(mu)
        
    # Sort Pareto indices by EI descending
    pareto_eis = ei[pareto_indices]
    sorted_pareto_indices = pareto_indices[np.argsort(pareto_eis)[::-1]]
    
    # Assign scores
    # "Proportional to their rank"
    # Rank 1 (best EI) -> Score 1.0
    # Rank 2 -> Score 0.9?
    # Let's use a simple linear decay or just rank/K.
    # If we use rank/K, the best gets 1.0, worst gets 1/K.
    # This ensures the best EI point is always preferred among Pareto points.
    
    scores = np.zeros_like(mu)
    num_pareto = len(sorted_pareto_indices)
    for rank, idx in enumerate(sorted_pareto_indices):
        # Rank 0 is the best (highest EI).
        # Score should be high for low rank index.
        # score = 1.0 - (rank / num_pareto) ?
        # Or simply: score = (num_pareto - rank) / num_pareto
        scores[idx] = (num_pareto - rank) / num_pareto
        
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
