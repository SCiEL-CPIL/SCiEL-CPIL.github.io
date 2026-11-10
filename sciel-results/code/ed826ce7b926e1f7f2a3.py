"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    TIME_LIMIT = 55  # seconds, leave buffer
    
    # Build conflict graph: conflict[i][j] = True if lcm(i,j) <= 256
    n = 256
    # vertices are 2..256, index 0..254
    # weight[i] = 1/(i+2)
    weights = [Fraction(1, i + 2) for i in range(n - 1)]
    num_vertices = n - 1  # 255 vertices
    
    # Build conflict adjacency
    # conflict[i] = set of j such that lcm(i+2, j+2) <= 256
    conflict = [set() for _ in range(num_vertices)]
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            a = i + 2
            b = j + 2
            if lcm(a, b) <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Order vertices by weight descending (i.e., by value ascending)
    # So vertex with value 2 comes first (highest weight)
    order = sorted(range(num_vertices), key=lambda i: weights[i], reverse=True)
    # order[0] is the vertex with value 2, order[1] is value 3, etc.
    
    # Map from original index to position in order
    pos_in_order = [0] * num_vertices
    for pos, idx in enumerate(order):
        pos_in_order[idx] = pos
    
    # Precompute suffix sums of weights for tighter bounds
    suffix_weight = [Fraction(0)] * (num_vertices + 1)
    for pos in range(num_vertices - 1, -1, -1):
        suffix_weight[pos] = suffix_weight[pos + 1] + weights[order[pos]]
    
    best_weight = Fraction(0)
    best_set = []
    
    # Branch and bound with better pruning
    def branch_and_bound(selected_set, current_weight, next_pos):
        nonlocal best_weight, best_set
        nonlocal start_time
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        # Upper bound: current_weight + sum of weights of all remaining vertices
        # that are not in conflict with any selected vertex
        ub = current_weight
        for pos in range(next_pos, num_vertices):
            idx = order[pos]
            # Check if idx conflicts with any selected vertex
            can_add = True
            for s in selected_set:
                if idx in conflict[s]:
                    can_add = False
                    break
            if can_add:
                ub += weights[idx]
        
        if ub <= best_weight:
            return
        
        # Try to include vertices from next_pos onwards
        for pos in range(next_pos, num_vertices):
            idx = order[pos]
            # Check if idx can be added (no conflict with selected)
            can_add = True
            for s in selected_set:
                if idx in conflict[s]:
                    can_add = False
                    break
            
            if can_add:
                # Include idx
                new_selected = selected_set | {idx}
                new_weight = current_weight + weights[idx]
                
                if new_weight > best_weight:
                    best_weight = new_weight
                    best_set = list(new_selected)
                
                # Recurse
                branch_and_bound(new_selected, new_weight, pos + 1)
                
                if time.time() - start_time > TIME_LIMIT:
                    return
            else:
                # Skip idx, continue
                continue
    
    # Initial call
    branch_and_bound(set(), Fraction(0), 0)
    
    # Convert back to values
    result = [idx + 2 for idx in best_set]
    return result
# EVOLVE-BLOCK-END
