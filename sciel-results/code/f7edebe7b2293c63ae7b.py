"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm

def construct():
    start_time = time.time()
    TIME_LIMIT = 55  # seconds, leave buffer
    
    n = 256
    weights = [1.0 / (i + 2) for i in range(n - 1)]
    num_vertices = n - 1  # 255 vertices
    
    # Build conflict adjacency
    conflict = [set() for _ in range(num_vertices)]
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            a = i + 2
            b = j + 2
            if lcm(a, b) <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Order vertices by weight descending (value ascending)
    order = sorted(range(num_vertices), key=lambda i: weights[i], reverse=True)
    
    # Map from original index to position in order
    pos_in_order = [0] * num_vertices
    for pos, idx in enumerate(order):
        pos_in_order[idx] = pos
    
    best_weight = 0.0
    best_set = []
    
    # Suffix sums for quick upper bound
    suffix_weights = [0.0] * (num_vertices + 1)
    for i in range(num_vertices - 1, -1, -1):
        suffix_weights[i] = suffix_weights[i + 1] + weights[order[i]]
    
    def branch_and_bound(selected_set, current_weight, next_pos):
        nonlocal best_weight, best_set
        nonlocal start_time
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        # Compute available vertices from next_pos onwards
        # and their total weight for a tighter bound
        available = []
        avail_weight = 0.0
        for pos in range(next_pos, num_vertices):
            idx = order[pos]
            can_add = True
            for s in selected_set:
                if idx in conflict[s]:
                    can_add = False
                    break
            if can_add:
                available.append(idx)
                avail_weight += weights[idx]
        
        ub = current_weight + avail_weight
        if ub <= best_weight:
            return
        
        # Try to include each available vertex
        for idx in available:
            new_selected = selected_set | {idx}
            new_weight = current_weight + weights[idx]
            
            if new_weight > best_weight:
                best_weight = new_weight
                best_set = list(new_selected)
            
            # Find the next position after idx in order
            next_pos_after = pos_in_order[idx] + 1
            
            # Recurse with remaining available vertices after this position
            # We need to pass the new selected set and continue from next_pos_after
            branch_and_bound(new_selected, new_weight, next_pos_after)
            
            if time.time() - start_time > TIME_LIMIT:
                return
    
    branch_and_bound(set(), 0.0, 0)
    
    result = [idx + 2 for idx in best_set]
    return result
# EVOLVE-BLOCK-END
