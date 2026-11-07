"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import product

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    
    # Adjacency as sets
    adj = [set() for _ in range(n)]
    for e in P['edges']:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    
    # Bitmask adjacency for fast clique/independent checks
    adj_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    deadline = time.time() + 55  # safety margin before 90s kill
    
    # Order vertices by degree descending (more constrained first)
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    order_idx = {v: i for i, v in enumerate(order)}
    
    def can_add_to_clique_mask(part_mask, v):
        """Check if v can be added to part as a clique using bitmask"""
        # v must be adjacent to all vertices in part_mask
        return (adj_mask[v] & part_mask) == part_mask
    
    def can_add_to_independent_mask(part_mask, v):
        """Check if v can be added to part as an independent set using bitmask"""
        # v must not be adjacent to any vertex in part_mask
        return (adj_mask[v] & part_mask) == 0
    
    best_parts = None
    best_count = n + 1  # worst case: n parts of size 1
    
    # Try to find a partition with k parts for k = 1, 2, ...
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        if k >= best_count:
            break
        
        # For k parts, each can be type 0 (clique) or 1 (independent)
        # Generate all type combinations, sorted by number of cliques (prefer more cliques for dense graphs)
        type_combos = list(product([0, 1], repeat=k))
        # Sort: prefer combinations with more cliques (type 0) first since graph is dense
        type_combos.sort(key=lambda tc: -sum(1 for t in tc if t == 0))
        
        found = False
        for types in type_combos:
            if time.time() > deadline:
                break
            
            # Symmetry breaking: for parts of the same type, enforce that earlier parts get vertices first
            # This reduces equivalent assignments
            parts = [0] * k  # bitmask for each part
            part_sizes = [0] * k
            
            # Precompute for each vertex and each part type whether it's feasible
            # and track which part indices of each type have been "used"
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == len(order):
                    return True
                v = order[idx]
                v_bit = 1 << v
                
                # Track last used part index for each type for symmetry breaking
                # We'll just try all parts but skip if a part of same type is empty and a later one isn't
                seen_empty_same_type = [False] * 2  # for type 0 and 1
                
                for part_idx in range(k):
                    t = types[part_idx]
                    
                    # Symmetry breaking: if this part is empty and there's an earlier empty part of same type, skip
                    if parts[part_idx] == 0:
                        if seen_empty_same_type[t]:
                            continue
                        seen_empty_same_type[t] = True
                    
                    if t == 0:  # clique
                        if can_add_to_clique_mask(parts[part_idx], v):
                            parts[part_idx] |= v_bit
                            part_sizes[part_idx] += 1
                            if backtrack(idx + 1):
                                return True
                            parts[part_idx] &= ~v_bit
                            part_sizes[part_idx] -= 1
                    else:  # independent
                        if can_add_to_independent_mask(parts[part_idx], v):
                            parts[part_idx] |= v_bit
                            part_sizes[part_idx] += 1
                            if backtrack(idx + 1):
                                return True
                            parts[part_idx] &= ~v_bit
                            part_sizes[part_idx] -= 1
                return False
            
            if backtrack(0):
                # Collect non-empty parts
                result = []
                for p in parts:
                    if p:
                        # Convert bitmask to list of vertices
                        verts = [v for v in range(n) if (p >> v) & 1]
                        result.append(verts)
                if result and len(result) < best_count:
                    best_parts = result
                    best_count = len(result)
                found = True
                break
        
        # If we found a solution with k parts, we're done (k is minimal)
        if found:
            break
    
    if best_parts is None:
        # Fallback: each vertex its own part
        best_parts = [[v] for v in range(n)]
    
    # Ensure all parts are non-empty and return as list of lists
    final = []
    for p in best_parts:
        if p:
            final.append(list(p))
    
    return final

# EVOLVE-BLOCK-END
