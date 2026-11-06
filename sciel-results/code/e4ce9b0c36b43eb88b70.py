import time
from math import gcd

def construct():
    start_time = time.time()
    MAX_TIME = 145.0  # Safety margin before 150s kill

    N = 511
    # Precompute values
    values = [0.0] * (N + 1)
    for i in range(1, N + 1):
        values[i] = 1.0 / (512.0 - i)
    
    # Build conflict graph: conflicts[a] = list of x where gcd(a, x) > 1
    conflicts = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for x in range(1, N + 1):
            if a != x and gcd(a, x) > 1:
                conflicts[a].append(x)
    
    # Sort vertices by value descending for branch-and-bound
    # Vertex 0 is unused; we'll work with 1-indexed
    vertices_by_value = list(range(1, N + 1))
    vertices_by_value.sort(key=lambda a: values[a], reverse=True)
    
    # Precompute a bitmask representation for fast independence checks
    # Use Python integers as bitsets (512 bits fits easily)
    # conflict_mask[a] has bit x set if a conflicts with x
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for x in conflicts[a]:
            mask |= (1 << x)
        conflict_mask[a] = mask
    
    # Greedy upper bound: sum of top k values where k is max independent set size estimate
    # For MWIS, a simple bound: if we have a candidate set, the upper bound is
    # current_weight + sum of values of vertices that could still be added
    # A tighter bound: use the fact that in an independent set, no two adjacent vertices can both be present
    
    best_set = []
    best_score = 0.0
    
    # Branch and bound for Maximum Weight Independent Set
    # We process vertices in order of decreasing weight
    # State: (index in sorted order, current_set_as_bitmask, current_weight)
    
    def branch_and_bound():
        nonlocal best_set, best_score
        
        # Upper bound function: given a bitmask of available vertices and current weight,
        # return an upper bound on the total weight achievable
        def upper_bound(available_mask, current_weight):
            # Simple bound: sum of all positive weights in available_mask
            # This is loose but fast
            total = current_weight
            m = available_mask
            while m:
                # Get lowest set bit
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                total += values[idx]
                m ^= lsb
            return total
        
        # Better bound: greedy coloring bound (Tomita's algorithm for MWIS)
        # For each vertex in available, compute a color class bound
        # This is more complex but much tighter
        
        # For now, use a simpler but effective bound:
        # Sort available vertices by weight, take them greedily maintaining independence,
        # sum their weights as the bound
        def greedy_bound(available_mask, current_weight):
            total = current_weight
            m = available_mask
            # Get vertices in available_mask sorted by weight
            verts = []
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                verts.append(idx)
                m ^= lsb
            verts.sort(key=lambda a: values[a], reverse=True)
            
            # Greedy: add vertices if independent with already added
            selected_mask = 0
            for v in verts:
                if not (selected_mask & conflict_mask[v]):
                    total += values[v]
                    selected_mask |= (1 << v)
            return total
        
        # Recursive B&B
        # We'll use an iterative approach with a stack to avoid recursion limits
        # Stack entries: (vertex_index_in_sorted_order, available_mask, current_weight, current_set_list)
        
        # Start: all vertices available, process from highest weight
        # We decide for each vertex (in weight order): include or exclude
        
        # For efficiency, we'll use a different formulation:
        # Process vertices in order. At each step, we can either include the vertex
        # (and remove its neighbors from available) or exclude it.
        
        # Let's use a cleaner B&B:
        # candidates: list of vertices still to consider (in weight order)
        # available: bitmask of vertices that can still be added (independent with chosen)
        
        def solve(candidates, available_mask, current_weight, current_set):
            nonlocal best_set, best_score
            
            if time.time() - start_time > MAX_TIME:
                return
            
            if not candidates:
                if current_weight > best_score:
                    best_score = current_weight
                    best_set = list(current_set)
                return
            
            # Upper bound check
            ub = greedy_bound(available_mask, current_weight)
            if ub <= best_score:
                return
            
            # Pick the next vertex to branch on
            # Choose the vertex with highest weight from candidates that is in available_mask
            # Actually, candidates should only contain vertices that are in available_mask
            # Let's filter candidates to only those in available_mask
            
            # Find the first vertex in candidates that is available
            v = -1
            for c in candidates:
                if available_mask & (1 << c):
                    v = c
                    break
            
            if v == -1:
                # No more available vertices in candidates
                if current_weight > best_score:
                    best_score = current_weight
                    best_set = list(current_set)
                return
            
            # Branch 1: Include v
            # Remove v and its neighbors from available
            new_available = available_mask & ~(conflict_mask[v] | (1 << v))
            # Remove v from candidates
            new_candidates = [c for c in candidates if c != v]
            
            solve(new_candidates, new_available, current_weight + values[v], current_set + [v])
            
            # Check time
            if time.time() - start_time > MAX_TIME:
                return
            
            # Branch 2: Exclude v
            new_candidates = [c for c in candidates if c != v]
            solve(new_candidates, available_mask, current_weight, current_set)
        
        # Initial call
        initial_candidates = list(vertices_by_value)
        initial_available = 0
        for v in range(1, N + 1):
            initial_available |= (1 << v)
        
        solve(initial_candidates, initial_available, 0.0, [])
    
    # Run branch and bound
    branch_and_bound()
    
    # If B&B didn't find anything or timed out early, use greedy as fallback
    if not best_set:
        # Simple greedy fallback
        chosen = []
        for a in range(N, 0, -1):
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        best_set = chosen
        best_score = sum(values[a] for a in best_set)
    
    return best_set
