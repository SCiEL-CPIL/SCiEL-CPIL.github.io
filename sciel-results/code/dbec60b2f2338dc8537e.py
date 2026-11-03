import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    D = 11
    
    # Generate all supports (combinations of 4 indices from 11)
    supports = list(itertools.combinations(range(D), 4))
    num_supports = len(supports)  # 330
    
    # Build conflict graph: two supports conflict if they share >= 3 indices
    # (i.e., their intersection size is >= 3)
    # Two weight-4 supports can share 0, 1, 2, 3, or 4 indices.
    # They conflict if they share >= 3 indices (i.e., share exactly 3 or 4).
    # If they share 4, they're the same support.
    # So conflicts are when intersection size is exactly 3.
    
    # Precompute for each pair of supports whether they conflict
    # Use bitmasks for efficient intersection computation
    support_masks = []
    for s in supports:
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        support_masks.append(mask)
    
    # Build adjacency list for the conflict graph
    # Node i conflicts with node j if |support_i ∩ support_j| >= 3
    # Since supports have size 4, intersection can be 0,1,2,3,4
    # Conflict if intersection is 3 or 4 (but 4 means same, so effectively 3)
    
    # For efficiency, precompute which supports share each index
    index_to_supports = [[] for _ in range(D)]
    for i, s in enumerate(supports):
        for idx in s:
            index_to_supports[idx].append(i)
    
    # Build conflict adjacency using bitsets
    # conflict[i] is a set of j such that i and j conflict
    conflict = [set() for _ in range(num_supports)]
    
    for i in range(num_supports):
        for j in range(i + 1, num_supports):
            # Check if intersection size >= 3
            inter = support_masks[i] & support_masks[j]
            # Count bits
            if bin(inter).count('1') >= 3:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Convert to sorted lists for faster iteration
    conflict_lists = [sorted(conflict[i]) for i in range(num_supports)]
    
    # Now solve Maximum Independent Set on this graph
    # Use branch and bound with greedy coloring upper bound
    
    # First, get an initial solution via greedy
    def greedy_mis():
        """Greedy algorithm for Maximum Independent Set"""
        remaining = set(range(num_supports))
        selected = []
        while remaining:
            # Pick node with minimum degree in remaining subgraph
            min_deg = float('inf')
            min_node = None
            for node in remaining:
                deg = len(conflict[node] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = node
            selected.append(min_node)
            remaining.discard(min_node)
            # Remove all neighbors of min_node from remaining
            remaining -= conflict[min_node]
        return selected
    
    initial_solution = greedy_mis()
    best_size = len(initial_solution)
    best_set = set(initial_solution)
    
    # Branch and bound with greedy coloring
    # For a set of candidate nodes, compute a greedy coloring to get upper bound
    def greedy_coloring(candidates):
        """
        Greedy coloring of the subgraph induced by candidates.
        Returns the number of colors used (upper bound on MIS size).
        """
        if not candidates:
            return 0
        
        candidates = sorted(candidates)
        colors = {}
        color_count = 0
        
        for node in candidates:
            # Find the smallest color not used by neighbors
            used_colors = set()
            for neighbor in conflict[node]:
                if neighbor in colors:
                    used_colors.add(colors[neighbor])
            
            c = 0
            while c in used_colors:
                c += 1
            
            colors[node] = c
            if c + 1 > color_count:
                color_count = c + 1
        
        return color_count
    
    # Branch and bound
    best_set_global = set(best_set)
    best_size_global = best_size
    deadline_check_interval = 1000
    nodes_explored = 0
    
    def branch_and_bound(candidates, current_set, current_size):
        nonlocal best_set_global, best_size_global, nodes_explored
        
        if time.time() > deadline:
            return
        
        # Prune: if current_size + upper_bound <= best, return
        upper_bound = greedy_coloring(candidates)
        if current_size + upper_bound <= best_size_global:
            return
        
        if not candidates:
            if current_size > best_size_global:
                best_size_global = current_size
                best_set_global = set(current_set)
            return
        
        # Pick a node to branch on
        # Use a heuristic: pick node with high degree in candidate subgraph
        # Or pick first node for simplicity
        node = candidates[0]
        
        # Branch 1: include node
        new_candidates = [c for c in candidates[1:] if c not in conflict[node]]
        new_set = current_set + [node]
        branch_and_bound(new_candidates, new_set, current_size + 1)
        
        # Branch 2: exclude node
        branch_and_bound(candidates[1:], current_set, current_size)
        
        nodes_explored += 1
        if nodes_explored % deadline_check_interval == 0 and time.time() > deadline:
            return
    
    # Run branch and bound
    all_nodes = list(range(num_supports))
    branch_and_bound(all_nodes, [], 0)
    
    # Use the best set found
    selected_supports = list(best_set_global)
    
    # Now generate all valid sign configurations for the selected supports
    # For each selected support, we can use any of the 16 sign patterns
    # but we need to check that no two vectors from different supports
    # violate the distance constraint.
    
    # Actually, the conflict graph already ensures that no two selected
    # supports share >= 3 indices. So for any two selected supports,
    # they share at most 2 indices. This means that for any sign patterns
    # on the two supports, the dot product is bounded.
    
    # Let's verify: if two supports share at most 2 indices, then for any
    # sign patterns, the dot product is at most 2 (in absolute value).
    # The norm squared of each vector is 4 (since 4 entries of ±1).
    # Distance squared = ||v1 - v2||^2 = ||v1||^2 + ||v2||^2 - 2*v1·v2
    # = 4 + 4 - 2*v1·v2 = 8 - 2*v1·v2
    # For distance >= max_norm = 2 (since norm = sqrt(4) = 2),
    # we need distance^2 >= 4, so 8 - 2*v1·v2 >= 4, i.e., v1·v2 <= 2.
    # Since max |v1·v2| = 2 (when sharing 2 indices with same signs),
    # this is satisfied.
    
    # Wait, let me reconsider. The task says "minimum pairwise distance is at least
    # the maximum vector norm". All vectors have the same norm (sqrt(4) = 2),
    # so max norm = 2. We need min distance >= 2.
    # Distance^2 = 8 - 2*v1·v2 >= 4 => v1·v2 <= 2.
    # Since |v1·v2| <= 2, this is always satisfied.
    
    # So we can use ALL 16 sign patterns for each selected support!
    
    # Generate all vectors
    vectors = []
    for sup_idx in selected_supports:
        s = supports[sup_idx]
        for cfg in range(16):
            v = np.zeros(D, dtype=np.int8)
            for i in range(4):
                sgn = 1 if ((cfg >> i) & 1) else -1
                v[s[i]] = sgn
            vectors.append(v)
    
    if not vectors:
        # Fallback
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for s_i in (-1, 1):
                    for s_j in (-1, 1):
                        v = np.zeros(D, dtype=np.int8)
                        v[i] = s_i
                        v[j] = s_j
                        vectors.append(v)
    
    return np.array(vectors, dtype=np.int8)
