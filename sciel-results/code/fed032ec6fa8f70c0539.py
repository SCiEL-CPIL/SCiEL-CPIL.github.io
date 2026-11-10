"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We keep a small cache to avoid recomputing the same fractional-knapsack
# surrogate too often, but the main design is:
#   1. Compute the optimal fractional knapsack solution (sort by density).
#   2. Identify the "critical" item (the one that is partially filled in the
#      fractional optimum).
#   3. Among items that are fully included in the fractional optimum, prefer
#      those that are "safe" (low weight, high value).
#   4. If no items are fully included, pick the highest-density feasible item.
#   5. Use a lookahead: for the top-k candidates, simulate adding them and
#      completing greedily by density to estimate total value.

# To keep things fast for 100-item instances with capacity 12.5, we do:
#   - Precompute density once per call (cheap: O(n log n)).
#   - The fractional solution tells us which items would be taken if we could
#     take fractions. The items with density above the critical density are
#     fully taken. The critical item is partially taken. Items below are not.
#   - Our heuristic: among the fully-taken items, pick the one with the best
#     "marginal contribution" considering that after picking it, we can still
#     fill the rest with the next-best density items.

# Actually, let's think more carefully. The key insight from the fractional
# knapsack is that the optimal discrete solution is close to the fractional
# optimum. The items that are fully included in the fractional solution are
# good candidates. Among those, we want to pick one that leaves the best
# remaining capacity for the rest.

# A simpler and effective approach: 
#   - Sort items by density descending.
#   - The fractional solution takes items in this order until capacity is
#     exhausted. The last item taken (partially) is the "break" item.
#   - For the discrete problem, we should consider the break item and its
#     neighbors, because the optimal discrete solution will include most of
#     the high-density items and may or may not include the break item.
#   - Heuristic: pick the item among the top-k (by density) that maximizes
#     the estimated total value if we pick it and then greedily fill the rest
#     by density.

# Let's implement this with a small k (e.g., k=5) for speed.

def _greedy_complete(remaining_capacity, remaining_weights, remaining_values):
    """Greedily fill remaining capacity by density, return total value."""
    if len(remaining_weights) == 0 or remaining_capacity <= 0:
        return 0.0
    # Filter feasible items
    mask = remaining_weights <= remaining_capacity
    if not np.any(mask):
        return 0.0
    w = remaining_weights[mask]
    v = remaining_values[mask]
    density = v / w
    order = np.argsort(-density)
    w = w[order]
    v = v[order]
    cap = remaining_capacity
    total = 0.0
    for wi, vi in zip(w, v):
        if wi <= cap:
            total += vi
            cap -= wi
        else:
            break
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter to feasible items (weight <= remaining capacity)
    mask = remaining_weights <= remaining_capacity
    if not np.any(mask):
        return -1
    
    idx_feasible = np.where(mask)[0]
    w = remaining_weights[idx_feasible]
    v = remaining_values[idx_feasible]
    
    if len(w) == 0:
        return -1
    
    # Compute density
    density = v / w
    
    # Sort by density descending
    order = np.argsort(-density)
    w_sorted = w[order]
    v_sorted = v[order]
    
    # Determine the fractional knapsack solution to find the "break" point
    cap = remaining_capacity
    break_idx = -1
    for i in range(len(w_sorted)):
        if w_sorted[i] <= cap:
            cap -= w_sorted[i]
        else:
            break_idx = i
            break
    
    # Candidates: the top-k items by density, plus the break item and its neighbors
    # We'll consider a small set of candidates and do a 1-step lookahead
    k = 8
    candidate_positions = set()
    
    # Top k by density
    for i in range(min(k, len(w_sorted))):
        candidate_positions.add(i)
    
    # Break item and neighbors
    if break_idx >= 0:
        for offset in range(-2, 3):
            j = break_idx + offset
            if 0 <= j < len(w_sorted):
                candidate_positions.add(j)
    
    # Convert to indices in the feasible array
    candidate_indices = [order[pos] for pos in sorted(candidate_positions)]
    
    # For each candidate, simulate picking it and greedily completing
    best_score = -1
    best_candidate = candidate_indices[0]
    
    for ci in candidate_indices:
        # Pick item ci
        new_cap = remaining_capacity - w[ci]
        if new_cap < 0:
            continue
        
        # Build remaining arrays without item ci
        mask_remove = np.ones(len(w), dtype=bool)
        mask_remove[ci] = False
        w_rem = w[mask_remove]
        v_rem = v[mask_remove]
        
        # Only keep feasible items
        mask_feas = w_rem <= new_cap
        if not np.any(mask_feas):
            score = v[ci]
        else:
            w_rem = w_rem[mask_feas]
            v_rem = v_rem[mask_feas]
            
            # Greedy complete
            density_rem = v_rem / w_rem
            order_rem = np.argsort(-density_rem)
            w_rem = w_rem[order_rem]
            v_rem = v_rem[order_rem]
            
            cap = new_cap
            total = v[ci]
            for wi, vi in zip(w_rem, v_rem):
                if wi <= cap:
                    total += vi
                    cap -= wi
                else:
                    break
            score = total
        
        if score > best_score:
            best_score = score
            best_candidate = ci
    
    # Return the index in the original remaining arrays
    return int(idx_feasible[best_candidate])
# EVOLVE-BLOCK-END
