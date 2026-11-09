import numpy as np
from scipy.stats import norm
import time

# Start time for the entire process? 
# The function is called repeatedly. We can't easily track "total" time across calls 
# unless we use a global variable or attribute. 
# However, the directive says "track elapsed time". 
# I will assume the "deadline" applies to the single call or the whole run. 
# Given the function is stateless, I will add a global start time.

_START_TIME = time.time()
_TIMEOUT = 320.0 # Safety margin before 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier filtering 
    and Hypervolume contribution scoring.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT:
        # Fallback: return something safe, e.g., lowest cost
        return -cost
    
    n = len(mu)
    
    # 1. Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # Avoid division by zero
    sigma_safe = np.where(sigma < 1e-9, 1e-9, sigma)
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If sigma is very small, this approximates max(0, f_best - mu)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # 2. Pareto Frontier Filter
    # We want to maximize EI and minimize Cost.
    # Sort indices by EI descending
    sort_idx = np.argsort(-ei)
    sorted_ei = ei[sort_idx]
    sorted_cost = cost[sort_idx]
    
    # A point is Pareto optimal if its cost is strictly lower than 
    # the minimum cost seen so far in the sorted list (which has higher or equal EI).
    # Initialize min_cost with infinity
    min_cost_so_far = np.full(n, np.inf)
    # Use cumulative minimum from the right? 
    # No, we iterate from highest EI to lowest.
    # If we sort by EI desc, then for index k, if sorted_cost[k] < min(sorted_cost[0:k]), 
    # it is not dominated by any point with higher EI.
    # We need to check if it is dominated by points with *same* EI but lower cost?
    # The definition of Pareto: A dominates B if A>=B in all objectives and > in at least one.
    # So if EI_A == EI_B and Cost_A < Cost_B, A dominates B.
    # So we should keep only the one with lowest cost for each EI level?
    # Actually, the cumulative minimum approach handles this:
    # If we have EI: [10, 10, 9] and Cost: [5, 3, 4]
    # Sorted by EI desc: 
    # idx 0: EI=10, Cost=5. Min cost so far (inf) -> 5 < inf. Pareto. Update min_cost=5.
    # idx 1: EI=10, Cost=3. Min cost so far (5) -> 3 < 5. Pareto. Update min_cost=3.
    # idx 2: EI=9, Cost=4. Min cost so far (3) -> 4 < 3? No. Dominated by idx 1 (EI 10 >= 9, Cost 3 <= 4).
    # This logic works.
    
    is_pareto = np.zeros(n, dtype=bool)
    current_min_cost = np.inf
    for i in range(n):
        idx = sort_idx[i]
        if sorted_cost[i] < current_min_cost:
            is_pareto[idx] = True
            current_min_cost = sorted_cost[i]
        else:
            # It is dominated by a point with higher (or equal) EI and lower (or equal) cost
            pass
            
    # 3. Hypervolume Contribution Scoring
    # Reference point: "origin" and "current best"?
    # Let's use the worst candidate in the entire set as the reference for normalization.
    # Ref_EI = min(ei)
    # Ref_Cost = max(cost)
    
    ei_min = np.min(ei)
    cost_max = np.max(cost)
    
    # Calculate scores for all points
    # Score = (EI - EI_min) * (Cost_max - Cost)
    scores = (ei - ei_min) * (cost_max - cost)
    
    # For non-Pareto points, we want to exclude them.
    # Set their scores to a very low value.
    scores[~is_pareto] = -1e6
    
    return scores
