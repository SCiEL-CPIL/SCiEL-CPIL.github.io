"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 150s limit

def _solve_dp(w, v, C, scale):
    """
    Solves 0/1 knapsack exactly using DP on scaled integer weights.
    Returns (list of indices in optimal solution, total value).
    """
    n = len(w)
    if n == 0:
        return [], 0.0
        
    cap_int = int(round(C * scale))
    if cap_int < 0:
        return [], 0.0

    # dp[i][c] = max value using first i items with capacity c
    # Using 1D DP for space efficiency and speed
    dp = np.zeros(cap_int + 1, dtype=np.float64)
    
    # We need to track which items were selected for backtracking
    # For 1D DP, we can either store parent pointers or just return the value
    # Since we only need the *next* item, and we are evaluating candidates,
    # we actually just need the optimal value of the subproblem.
    # The current program computes the full optimal value for the subproblem.
    # Let's optimize this to just return the max value.
    
    for i in range(n):
        wi = int(round(w[i] * scale))
        vi = v[i]
        if wi > cap_int:
            continue
        # Update in reverse
        # dp[c] = max(dp[c], dp[c-wi] + vi)
        # Vectorized update:
        # candidate = dp[:cap_int-wi+1] + vi
        # dp[wi:] = np.maximum(dp[wi:], candidate)
        
        # Note: This vectorized update is incorrect for 0/1 knapsack if done naively in-place
        # because dp[c-wi] might have been updated in the same iteration.
        # However, if we iterate c from cap_int down to wi, it works.
        # Vectorization of the reverse loop is tricky without creating a copy.
        # For cap_int ~ 1250 and n ~ 100, a simple loop is fast enough.
        # But let's try to be efficient.
        
        # Standard 1D DP loop
        # We can't easily vectorize the dependency dp[c] depends on dp[c-wi] (old value)
        # unless we use a temporary array or iterate carefully.
        # Given small sizes, Python loop is acceptable, but let's check performance.
        # 100 items * 1250 capacity = 125,000 operations. Very fast.
        
        for c in range(cap_int, wi - 1, -1):
            candidate_val = dp[c - wi] + vi
            if candidate_val > dp[c]:
                dp[c] = candidate_val
                
    return dp[cap_int]

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add.
    Returns an index into the remaining (feasible) arrays, or -1 to stop.
    
    Strategy: 
    1. Check deadline.
    2. Use Lagrangian relaxation to identify promising candidates.
       Find lambda such that sum(w_i * 1{v_i > lambda*w_i}) ~ remaining_capacity.
       This effectively solves the fractional knapsack problem and identifies
       the "break item" and items that should definitely be in or out.
    3. Use exact DP on a reduced set of candidates (top K by Lagrangian score)
       to make the final 1-step look-ahead decision.
    """
    global _start_time
    
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Fallback: Simple greedy by density
        if len(remaining_weights) == 0:
            return -1
        w = np.array(remaining_weights)
        v = np.array(remaining_values)
        feasible = np.where(w <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v[feasible] / w[feasible]
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
        
    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)
    C = remaining_capacity
    
    # Filter to feasible items
    feasible_mask = w <= C
    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 0:
        return -1
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # Scale factor
    scale = 100
    cap_int = int(round(C * scale))
    
    # If capacity is too large for DP, fall back to density
    if cap_int > 50000:
        densities = v / w
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        return int(np.argmax(densities[feasible_mask]))
    
    # Lagrangian Relaxation to find promising candidates
    # We want to find lambda such that the weight of items with v_i/w_i > lambda
    # is close to C.
    # This is equivalent to finding the break item in fractional knapsack.
    
    # Calculate densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
    
    # Sort by density descending
    sorted_indices = np.argsort(-densities, kind='stable')
    
    # Find the break item
    current_weight = 0.0
    break_idx = -1
    for idx in sorted_indices:
        if w[idx] > C - current_weight:
            break_idx = idx
            break
        current_weight += w[idx]
    
    # The optimal fractional solution takes all items before break_idx,
    # a fraction of break_idx, and none after.
    # For 0/1, the optimal solution is likely to contain most items before break_idx
    # and possibly some items around break_idx.
    
    # Candidates: 
    # 1. The break item itself
    # 2. Items just before and after the break item (neighbors in density order)
    # 3. Top K items by density
    
    # Let's select a set of candidates to evaluate with DP
    # We want to keep the set small for speed, but broad enough to not miss the optimal next item.
    
    # Strategy: 
    # - Always include the break item if it exists.
    # - Include top 10 items by density.
    # - Include bottom 5 items by density that are still feasible? No, unlikely to be useful.
    # - Actually, the "next item" in a sequential build-up might not be the one in the global optimal set.
    #   But the 1-step look-ahead DP handles this: it evaluates "if I pick X, what is the best completion?".
    #   So we just need a good set of X candidates.
    
    # Let's take the top 20 items by density as candidates.
    # Also, if break_idx is found, ensure it is in the candidates.
    
    top_k = min(20, len(sorted_indices))
    candidate_set = set(sorted_indices[:top_k].tolist())
    
    if break_idx != -1:
        candidate_set.add(break_idx)
        
    # Also add a few items around the break point if not already included
    if break_idx != -1:
        pos_in_sorted = np.where(sorted_indices == break_idx)[0]
        if len(pos_in_sorted) > 0:
            pos = pos_in_sorted[0]
            for delta in [-2, -1, 1, 2]:
                new_pos = pos + delta
                if 0 <= new_pos < len(sorted_indices):
                    candidate_set.add(sorted_indices[new_pos])
    
    candidates = np.array(list(candidate_set))
    
    # Ensure candidates are feasible (they should be, but just in case)
    candidates = candidates[w[candidates] <= C]
    
    if len(candidates) == 0:
        return -1
        
    # For each candidate, compute the optimal value of the subproblem
    # using the remaining items (excluding the candidate) and remaining capacity.
    
    best_score = -1
    best_idx = int(candidates[0])
    
    # Pre-compute scaled weights
    w_scaled = (w * scale).astype(np.int64)
    
    for cand in candidates:
        cand_w_int = w_scaled[cand]
        cand_val = v[cand]
        
        remaining_cap = cap_int - cand_w_int
        if remaining_cap < 0:
            continue
            
        # Build subproblem: all items except cand
        # We can just pass the full arrays and mask out cand?
        # Or create new arrays. Creating new arrays is O(n).
        # n is ~100, so it's fine.
        
        mask = np.ones(n, dtype=bool)
        mask[cand] = False
        sub_indices = np.where(mask)[0]
        
        if len(sub_indices) == 0:
            total = cand_val
        else:
            sub_w = w[sub_indices]
            sub_v = v[sub_indices]
            # Solve DP for subproblem
            # We only need the max value, not the indices
            sub_opt_val = _solve_dp(sub_w, sub_v, remaining_cap, scale)
            total = cand_val + sub_opt_val
            
        if total > best_score:
            best_score = total
            best_idx = int(cand)
            
    return best_idx
# EVOLVE-BLOCK-END
