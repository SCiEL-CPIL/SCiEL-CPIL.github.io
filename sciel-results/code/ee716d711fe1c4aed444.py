"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 512
    # We want to select pairwise coprime integers a in [1, 511]
    # maximizing sum of 1/(512-a)
    
    # Get a good initial solution with greedy
    def greedy_solution():
        chosen = []
        for a in range(511, 0, -1):
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    best_solution = greedy_solution()
    best_score = sum(1.0 / (n - a) for a in best_solution)
    
    # Local search polish
    def local_search(solution, max_iterations=500):
        solution = list(solution)
        current_score = sum(1.0 / (n - a) for a in solution)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline:
                break
            for i, a in enumerate(solution):
                remaining = [x for j, x in enumerate(solution) if j != i]
                for b in range(511, 0, -1):
                    if b == a or b in remaining:
                        continue
                    if 1.0 / (n - b) <= 1.0 / (n - a):
                        break
                    if all(gcd(b, x) == 1 for x in remaining):
                        solution = remaining + [b]
                        current_score += 1.0 / (n - b) - 1.0 / (n - a)
                        improved = True
                        break
                if improved:
                    break
        return solution, current_score
    
    best_solution, best_score = local_search(best_solution)
    
    # Branch and bound with bitset conflict tracking
    # Vertices: integers 1..511, edge if gcd > 1
    # Use bitmasks for conflict sets (511 bits fits in Python int)
    
    # Precompute conflict masks: for each number, which numbers conflict with it
    conflict_masks = [0] * 512  # index 1..511
    for a in range(1, 512):
        mask = 0
        for b in range(1, 512):
            if a != b and gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_masks[a] = mask
    
    # Weights sorted descending: a=511 has weight 1/1, a=510 has 1/2, etc.
    # Order vertices by weight descending (a descending)
    vertex_order = list(range(511, 0, -1))
    weights = {a: 1.0 / (n - a) for a in range(1, 512)}
    
    # For B&B, we process vertices in weight order
    # State: (next_vertex_index, available_mask, current_score, current_set)
    # available_mask: bitmask of vertices that can still be added
    
    # Upper bound: sum of weights of all available vertices (relaxation)
    # Better bound: greedy upper bound on the available set
    
    # Precompute prefix sums of weights in vertex_order for quick upper bounds
    prefix_weight = [0.0] * (len(vertex_order) + 1)
    for i, v in enumerate(vertex_order):
        prefix_weight[i + 1] = prefix_weight[i] + weights[v]
    
    # Greedy upper bound: greedily select from available vertices in weight order
    def greedy_upper_bound(available_mask, start_idx=0):
        ub = 0.0
        for i in range(start_idx, len(vertex_order)):
            if time.time() > deadline:
                return ub
            v = vertex_order[i]
            if (available_mask >> v) & 1:
                ub += weights[v]
                # Remove v and all its conflicts from available
                available_mask &= ~(conflict_masks[v] | (1 << v))
        return ub
    
    # Branch and bound
    best_score_bb = best_score
    best_set_bb = set(best_solution)
    
    def branch_and_bound(available_mask, start_idx, current_score, current_set):
        nonlocal best_score_bb, best_set_bb
        
        if time.time() > deadline:
            return
        
        # If no more vertices to consider
        if start_idx >= len(vertex_order):
            if current_score > best_score_bb:
                best_score_bb = current_score
                best_set_bb = set(current_set)
            return
        
        # Upper bound: current + greedy UB on remaining available
        ub = current_score + greedy_upper_bound(available_mask, start_idx)
        if ub <= best_score_bb:
            return
        
        # Try to include vertex_order[start_idx]
        v = vertex_order[start_idx]
        if (available_mask >> v) & 1:
            # Include v
            new_set = current_set | {v}
            new_score = current_score + weights[v]
            new_mask = available_mask & ~(conflict_masks[v] | (1 << v))
            branch_and_bound(new_mask, start_idx + 1, new_score, new_set)
            
            # Exclude v
            branch_and_bound(available_mask, start_idx + 1, current_score, current_set)
        else:
            # v not available, skip
            branch_and_bound(available_mask, start_idx + 1, current_score, current_set)
    
    # Start B&B from the full problem
    initial_mask = 0
    for a in range(1, 512):
        initial_mask |= (1 << a)
    
    # But B&B on all 511 vertices might be too slow. Let's limit depth or use a cutoff.
    # Actually, let's try a limited B&B that only explores promising branches.
    # We'll use a node limit.
    
    node_count = [0]
    max_nodes = 100000
    
    def branch_and_bound_limited(available_mask, start_idx, current_score, current_set):
        nonlocal best_score_bb, best_set_bb
        
        if time.time() > deadline:
            return
        if node_count[0] >= max_nodes:
            return
        node_count[0] += 1
        
        if start_idx >= len(vertex_order):
            if current_score > best_score_bb:
                best_score_bb = current_score
                best_set_bb = set(current_set)
            return
        
        # Upper bound
        ub = current_score + greedy_upper_bound(available_mask, start_idx)
        if ub <= best_score_bb:
            return
        
        v = vertex_order[start_idx]
        if (available_mask >> v) & 1:
            # Include v
            new_set = current_set | {v}
            new_score = current_score + weights[v]
            new_mask = available_mask & ~(conflict_masks[v] | (1 << v))
            branch_and_bound_limited(new_mask, start_idx + 1, new_score, new_set)
            
            # Exclude v
            branch_and_bound_limited(available_mask, start_idx + 1, current_score, current_set)
        else:
            branch_and_bound_limited(available_mask, start_idx + 1, current_score, current_set)
    
    if time.time() < deadline - 5:
        branch_and_bound_limited(initial_mask, 0, 0.0, set())
    
    if best_score_bb > best_score:
        best_score = best_score_bb
        best_solution = list(best_set_bb)
    
    # Final local search polish
    if time.time() < deadline - 2:
        final_sol, final_score = local_search(best_solution, max_iterations=300)
        if final_score > best_score:
            best_score = final_score
            best_solution = final_sol
    
    return best_solution
# EVOLVE-BLOCK-END
