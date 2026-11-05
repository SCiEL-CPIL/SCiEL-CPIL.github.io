"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if len(remaining_weights) == 0:
        return -1

    start_time = time.time()
    # Deadline guard: 120s total for 8 instances * ~100 calls = 800 calls.
    # 120/800 = 0.15s per call. Use 0.12s to be safe.
    deadline = start_time + 0.12

    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    
    # Filter feasible items
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # Lagrangian Relaxation Heuristic:
    # We want to select an item that maximizes projected value.
    # Instead of simulating greedy for every item (O(N^2)), we use a Lagrangian multiplier lambda
    # to score items: score_i = v_i - lambda * w_i.
    # The optimal lambda for the fractional knapsack problem is the density of the item
    # that exactly fills the capacity (or the last item in the greedy sort).
    # For 0/1, we approximate this.
    
    # 1. Sort items by density (value/weight) descending.
    # Handle zero weights: assign infinite density.
    densities = np.zeros(n)
    non_zero = weights > 0
    densities[non_zero] = values[non_zero] / weights[non_zero]
    densities[~non_zero] = float('inf')
    
    sorted_indices = np.argsort(-densities)
    
    # 2. Find the "critical" density (lambda) by simulating the fractional knapsack.
    # This lambda represents the marginal value of capacity.
    current_cap = remaining_capacity
    lambda_val = 0.0
    for idx in sorted_indices:
        w = weights[idx]
        v = values[idx]
        if w <= current_cap:
            current_cap -= w
            lambda_val = v / w if w > 0 else float('inf')
        else:
            if w > 0:
                lambda_val = v / w
            break
    if current_cap > 0:
        lambda_val = 0.0 # If we can take all, marginal value is 0? Or low.
        
    # 3. Score all feasible items using v_i - lambda * w_i.
    # This score approximates the net gain in total value if we were to swap capacity allocation.
    # Items with high v_i - lambda * w_i are "better than average" relative to the current marginal rate.
    
    best_index = -1
    best_score = -float('inf')
    
    for i_idx in feasible_indices:
        if time.time() > deadline:
            # Fallback: pick highest density feasible item
            return int(sorted_indices[0]) if sorted_indices[0] in feasible_indices else int(feasible_indices[0])
            
        w_i = weights[i_idx]
        v_i = values[i_idx]
        
        # Lagrangian score
        score = v_i - lambda_val * w_i
        
        # Tie-breaking: if scores are close, prefer higher density or higher value?
        # Let's add a small bonus for density to break ties in favor of more efficient items
        # if the Lagrangian score is very close.
        if score > best_score:
            best_score = score
            best_index = int(i_idx)
            
    if best_index == -1:
        return -1
        
    return best_index

# EVOLVE-BLOCK-END
