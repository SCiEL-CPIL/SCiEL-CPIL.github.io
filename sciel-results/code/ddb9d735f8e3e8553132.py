"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    n = 256
    
    # Build conflict graph: vertices are 2..256, edge if not coprime
    # We'll use bitmasks for efficiency
    # Number of vertices: 255 (numbers 2 through 256)
    num_v = n - 1  # 255 vertices, indexed 0..254, vertex i = number i+2
    
    # Precompute coprime relationships using bitmasks
    # coprime_mask[i] has bit j set if vertex i and vertex j are coprime (no edge)
    coprime_mask = [0] * num_v
    for i in range(num_v):
        val_i = i + 2
        mask = 0
        for j in range(num_v):
            if i == j:
                continue
            val_j = j + 2
            if gcd(val_i, val_j) == 1:
                mask |= (1 << j)
        coprime_mask[i] = mask
    
    # Weight of each vertex = its value
    weights = [i + 2 for i in range(num_v)]
    
    # Greedy coloring for upper bound
    # Colors are assigned so that vertices in the same color are independent
    # Upper bound = sum of max weight in each color class
    
    def greedy_color_bound(candidates_mask):
        """Return upper bound on sum of weights of independent set from candidates."""
        # Greedy coloring: repeatedly pick a vertex, assign it a color,
        # remove it and its non-coprime neighbors from consideration for that color
        remaining = candidates_mask
        bound = 0
        while remaining:
            # Pick vertex with highest weight from remaining
            # Find highest weight vertex in remaining
            best_idx = -1
            best_w = -1
            temp = remaining
            while temp:
                # Get lowest set bit
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                w = weights[idx]
                if w > best_w:
                    best_w = w
                    best_idx = idx
                temp &= temp - 1
            
            if best_idx < 0:
                break
            
            bound += best_w
            
            # For this color class, we can only include vertices coprime to best_idx
            # among those still in remaining
            color_class_candidates = remaining & coprime_mask[best_idx]
            
            # Remove best_idx and all its non-coprime neighbors from remaining
            # Actually for coloring bound: we remove best_idx, then for next color
            # we consider remaining minus (best_idx and its neighbors in the conflict graph)
            # Wait - greedy coloring for independent set bound:
            # We partition vertices into color classes where each class is an independent set
            # Upper bound = sum of (max weight in each color class)
            # To form color classes: pick a vertex, it forms one color with all its non-neighbors
            # But that's not quite right either.
            
            # Simpler approach: sequential coloring
            # Assign color 1 to a vertex, then color 2 to next, etc.
            # Upper bound = sum over colors of max weight in that color
            # This is valid if each color class is an independent set.
            
            # Let me use a different approach: 
            # Upper bound via greedy sequential selection
            # Pick highest weight vertex, add to bound, remove it and all its neighbors
            # This gives a valid upper bound (it's actually a lower bound on the max independent set,
            # but as an upper bound for pruning, we need something else)
            
            # Actually for branch and bound on MWIS:
            # Upper bound = current_weight + sum of weights of all remaining candidates
            # That's too loose.
            
            # Better: use the fact that in any independent set, at most one vertex from each
            # conflicting pair can be chosen.
            
            # Let me just use: upper bound = current + sum of all remaining weights
            # It's loose but simple. Or use coloring properly.
            
            # Proper greedy coloring bound:
            # Color vertices sequentially. Each color class must be an independent set.
            # Upper bound = sum of (max weight in each color class).
            
            # For simplicity and speed, let me use a simpler bound:
            # upper_bound = current + sum of weights of remaining vertices
            # This is always valid but may not prune well.
            
            # Let me implement proper greedy coloring:
            break  # Skip to simpler bound for now
        
        # Use simple bound: sum of all remaining weights
        temp = candidates_mask
        s = 0
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            s += weights[idx]
            temp &= temp - 1
        return s
    
    # Branch and bound for maximum weight independent set
    best_weight = [0]
    best_set = [set()]
    
    def branch_and_bound(candidates_mask, current_weight, current_set):
        if time.time() > deadline:
            return
        
        if candidates_mask == 0:
            if current_weight > best_weight[0]:
                best_weight[0] = current_weight
                best_set[0] = set(current_set)
            return
        
        # Upper bound: current_weight + sum of all remaining weights
        # Compute sum of remaining weights
        temp = candidates_mask
        rem_sum = 0
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            rem_sum += weights[idx]
            temp &= temp - 1
        
        if current_weight + rem_sum <= best_weight[0]:
            return
        
        # Pick a vertex to branch on
        # Choose vertex with highest weight from candidates
        temp = candidates_mask
        best_idx = -1
        best_w = -1
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            w = weights[idx]
            if w > best_w:
                best_w = w
                best_idx = idx
            temp &= temp - 1
        
        if best_idx < 0:
            return
        
        # Branch 1: Include best_idx
        new_set = current_set | {best_idx}
        # New candidates: those coprime to best_idx and still in candidates
        new_candidates = candidates_mask & coprime_mask[best_idx]
        # Remove best_idx from new_candidates (it's already excluded by coprime_mask since i!=j check)
        # Actually coprime_mask[best_idx] doesn't include best_idx itself, so it's fine
        branch_and_bound(new_candidates, current_weight + best_w, new_set)
        
        # Branch 2: Exclude best_idx
        new_candidates = candidates_mask & ~(1 << best_idx)
        branch_and_bound(new_candidates, current_weight, current_set)
    
    # Start with all vertices as candidates
    all_mask = (1 << num_v) - 1
    branch_and_bound(all_mask, 0, set())
    
    # Convert best set to list of integers
    result = [1]  # Always include 1
    for idx in best_set[0]:
        result.append(idx + 2)
    
    # Ensure all are Python ints and return as list
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
