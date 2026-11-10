# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    D = 15
    N = 10

    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]

    # Precompute path masks and weights
    # We will sort demands by weight descending to improve B&B pruning
    # Original indices are needed to return the correct array order
    original_indices = np.arange(D)
    
    # Sort demands by weight descending
    sorted_indices = np.argsort([d[2] for d in demands])[::-1]
    sorted_demands = [demands[i] for i in sorted_indices]
    
    weights = np.array([d[2] for d in sorted_demands], dtype=float)
    
    # Precompute delta vectors: change in load if we switch from CW to CCW
    # delta[i] = w_i * (ccw_mask[i] - cw_mask[i])
    # Note: cw_mask + ccw_mask = 1 (for all edges, as they partition the ring)
    # So ccw_mask = 1 - cw_mask
    # delta[i] = w_i * (1 - cw_mask[i] - cw_mask[i]) = w_i * (1 - 2*cw_mask[i])
    
    deltas = np.zeros((D, N), dtype=float)
    min_path_lengths = np.zeros(D, dtype=float)
    
    for i, (s, t, w) in enumerate(sorted_demands):
        cw_mask = np.zeros(N, dtype=float)
        e = s
        length = 0
        while e != t:
            cw_mask[e] = 1.0
            e = (e + 1) % N
            length += 1
        
        # ccw_mask is 1 - cw_mask
        # delta = w * (ccw_mask - cw_mask) = w * (1 - 2*cw_mask)
        deltas[i] = w * (1.0 - 2.0 * cw_mask)
        
        # Min path length for lower bound calculation
        min_path_lengths[i] = min(length, N - length)

    # Initial state: All demands routed CW
    # Load for all CW:
    # Load[e] = sum(w_i * cw_mask[i, e])
    # We can compute this efficiently.
    
    # Let's use a recursive Branch and Bound
    # State: current_loads (array 10), idx (next demand index), current_choices (array D)
    
    best_load = float('inf')
    best_choices = np.zeros(D, dtype=int) # 0 for CW, 1 for CCW in sorted order
    
    # Deadline guard
    deadline = time.time() + 590
    
    # Precompute prefix sums of min_path_lengths for the LB sum calculation?
    # LB_sum = (Current_Sum_Loas + sum_{j=idx}^{D-1} w_j * min_len_j) / N
    # We need Current_Sum_Loads.
    
    # To speed up, we can maintain current_loads and current_sum_loads.
    
    def solve(idx, current_loads, current_sum_loads, choices):
        nonlocal best_load, best_choices
        
        if time.time() > deadline:
            return
        
        if idx == D:
            max_load = np.max(current_loads)
            if max_load < best_load:
                best_load = max_load
                best_choices = choices.copy()
            return
        
        # Lower Bound 1: Max of current loads
        lb1 = np.max(current_loads)
        
        # Lower Bound 2: Sum based
        # Minimum additional sum load from remaining demands
        # rem_min_sum = sum(weights[j] * min_path_lengths[j] for j in idx..D-1)
        # We can precompute suffix sums of weights * min_path_lengths
        rem_min_sum = suffix_min_sum[idx]
        total_min_sum = current_sum_loads + rem_min_sum
        lb2 = total_min_sum / N
        
        lb = max(lb1, lb2)
        
        if lb >= best_load:
            return
        
        # Branch on demand idx
        # Option 1: Route CW (choice 0)
        # Load change: 0 (relative to the "all CW" baseline? No, we are building up)
        # Wait, the deltas are defined as change from CW to CCW.
        # If we start with "all CW", load is fixed.
        # Here, we are building the load from scratch.
        # Let's redefine:
        # Start with load = 0.
        # If we choose CW for demand i, we add w_i * cw_mask[i].
        # If we choose CCW for demand i, we add w_i * ccw_mask[i].
        
        # Let's stick to the delta approach from a baseline.
        # Baseline: All CW.
        # Base_Load = sum(w_i * cw_mask[i])
        # If we choose CCW for demand i, we add delta[i] to Base_Load.
        
        # So, in the recursion, let's assume we are deciding the "deviations" from All-CW.
        # current_loads represents Base_Load + sum(deltas[j] for j decided as CCW).
        
        # Initialize outside:
        # base_loads = (cw_matrix.T @ weights)
        # base_sum = np.sum(base_loads)
        
        # In solve(idx, ...):
        # current_loads is the load vector.
        
        # Branch 1: Choose CW for demand idx (no change to loads from base)
        # choices[idx] = 0
        solve(idx + 1, current_loads, current_sum_loads, choices)
        
        # Branch 2: Choose CCW for demand idx (add delta[idx])
        new_loads = current_loads + deltas[idx]
        new_sum_loads = current_sum_loads + np.sum(deltas[idx])
        choices[idx] = 1
        solve(idx + 1, new_loads, new_sum_loads, choices)
        choices[idx] = 0 # backtrack

    # Precompute base loads (all CW)
    cw_matrix = np.zeros((D, N), dtype=float)
    for i, (s, t, w) in enumerate(sorted_demands):
        e = s
        while e != t:
            cw_matrix[i, e] = 1.0
            e = (e + 1) % N
            
    base_loads = cw_matrix.T @ weights
    base_sum_loads = np.sum(base_loads)
    
    # Precompute suffix sums of (w * min_len)
    # suffix_min_sum[i] = sum_{j=i}^{D-1} weights[j] * min_path_lengths[j]
    suffix_min_sum = np.zeros(D + 1)
    for i in range(D - 1, -1, -1):
        suffix_min_sum[i] = suffix_min_sum[i + 1] + weights[i] * min_path_lengths[i]

    # Start recursion
    # We need to pass base_loads and base_sum_loads
    # But solve() signature above doesn't take base.
    # I will modify solve to take the current state relative to base.
    # Actually, the code above assumes current_loads is the absolute load.
    # So I should pass base_loads as the initial current_loads.
    
    # However, the recursive function `solve` defined above uses `suffix_min_sum` which is global.
    # And it uses `deltas` which is global.
    # It modifies `choices`.
    
    # Let's refine the solve function to be a local closure or pass parameters.
    
    def _solve(idx, current_loads, current_sum_loads, choices):
        nonlocal best_load, best_choices
        
        if time.time() > deadline:
            return
        
        if idx == D:
            max_load = np.max(current_loads)
            if max_load < best_load:
                best_load = max_load
                best_choices = choices.copy()
            return
        
        # Lower Bound
        lb1 = np.max(current_loads)
        lb2 = (current_sum_loads + suffix_min_sum[idx]) / N
        lb = max(lb1, lb2)
        
        if lb >= best_load:
            return
        
        # Branch 1: CW (no change)
        _solve(idx + 1, current_loads, current_sum_loads, choices)
        
        # Branch 2: CCW (add delta)
        # Optimization: Check if adding delta improves things? 
        # No, just branch.
        new_loads = current_loads + deltas[idx]
        new_sum_loads = current_sum_loads + np.sum(deltas[idx])
        choices[idx] = 1
        _solve(idx + 1, new_loads, new_sum_loads, choices)
        choices[idx] = 0

    _solve(0, base_loads, base_sum_loads, np.zeros(D, dtype=int))

    # Map sorted choices back to original order
    final_routing = np.zeros(D, dtype=int)
    for orig_idx, sorted_idx in enumerate(sorted_indices):
        final_routing[orig_idx] = best_choices[sorted_idx]
        
    return final_routing

# EVOLVE-BLOCK-END
