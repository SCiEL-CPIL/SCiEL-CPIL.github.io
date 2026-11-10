"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    from math import gcd, isqrt
    
    deadline = time.time() + 55  # 55s safety margin
    
    N = 512
    # Build conflict graph: edge between i,j if lcm(i,j) > 512
    # We want maximum independent set in this graph = maximum clique in complement
    
    # For efficiency, use bitsets (Python ints as bitsets)
    # Vertex i (0-indexed) corresponds to number i+1
    
    # Precompute compatibility
    # compat[i] is a bitmask where bit j is 1 if vertex i is compatible with vertex j
    # (i.e., lcm(i+1, j+1) <= 512)
    
    n = N
    # compat[i] has bit j set if (i+1, j+1) are compatible
    compat = [0] * n
    for i in range(n):
        a = i + 1
        mask = 0
        for j in range(n):
            b = j + 1
            # lcm(a,b) = a*b // gcd(a,b)
            g = gcd(a, b)
            lcm_val = a * b // g
            if lcm_val <= 512:
                mask |= (1 << j)
        compat[i] = mask
    
    # Maximum clique in compatibility graph (where edge means compatible)
    # Using branch and bound with coloring heuristic
    
    # Order vertices by degree (descending) for better pruning
    degrees = [bin(compat[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Remap: new index k corresponds to old index order[k]
    # Build remapped compat
    remap = [0] * n
    for k, old_idx in enumerate(order):
        remap[old_idx] = k
    
    remapped_compat = [0] * n
    for old_i in range(n):
        new_i = remap[old_i]
        mask = 0
        old_mask = compat[old_i]
        for old_j in range(n):
            if old_mask & (1 << old_j):
                new_j = remap[old_j]
                mask |= (1 << new_j)
        remapped_compat[new_i] = mask
    
    # Now find maximum clique using branch and bound
    # Standard Tomita algorithm with greedy coloring
    
    best_clique = []
    best_size = 0
    best_set = set()
    
    def color_sort(candidates):
        """Color the candidate vertices greedily for upper bound."""
        # candidates is a set/list of vertex indices
        # Returns list of (vertex, color) sorted by color
        cand_list = list(candidates)
        colors = {}
        color = 0
        uncolored = set(cand_list)
        while uncolored:
            color += 1
            # Greedily assign this color to independent set
            available = set(uncolored)
            while available:
                # Pick vertex with highest degree in remaining
                v = max(available, key=lambda x: len([y for y in available if x != y and (remapped_compat[x] >> y) & 1]))
                colors[v] = color
                uncolored.discard(v)
                available.discard(v)
                # Remove neighbors of v from available
                to_remove = set()
                for y in list(available):
                    if (remapped_compat[v] >> y) & 1:
                        to_remove.add(y)
                available -= to_remove
        # Sort by color
        sorted_cands = sorted(colors.keys(), key=lambda x: colors[x])
        return [(v, colors[v]) for v in sorted_cands]
    
    def expand(candidates, current_clique, depth):
        nonlocal best_clique, best_size, best_set
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = list(current_clique)
                best_set = set(current_clique)
            return
        
        # Get upper bound from coloring
        colored = color_sort(candidates)
        
        # Process in reverse color order
        for i in range(len(colored) - 1, -1, -1):
            v, c = colored[i]
            
            if time.time() > deadline:
                return
            
            # Upper bound: current_clique size + c
            if len(current_clique) + c <= best_size:
                break  # All remaining have <= color, so prune
            
            # Add v to clique
            new_candidates = set()
            for u in candidates:
                if u != v and (remapped_compat[v] >> u) & 1:
                    new_candidates.add(u)
            
            current_clique.append(v)
            expand(new_candidates, current_clique, depth + 1)
            current_clique.pop()
            
            candidates.discard(v)
    
    # Initial candidates: all vertices
    initial_cands = set(range(n))
    expand(initial_cands, [], 0)
    
    # Convert back to original numbers
    if best_set:
        result = [order[v] + 1 for v in best_set]
        result.sort()
        return result
    else:
        return [1]

# EVOLVE-BLOCK-END
