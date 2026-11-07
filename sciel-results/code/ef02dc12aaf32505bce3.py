"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask for each vertex
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    size = 1 << n
    full_mask = size - 1
    
    # Precompute is_clique and is_indep using bit-parallel checks
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
        if is_indep[rest] and (rest & adj[v]) == 0:
            is_indep[mask] = True
    
    valid = [is_clique[m] or is_indep[m] for m in range(size)]
    
    # Find all maximal homogeneous sets (maximal cliques and maximal independent sets)
    # A set is maximal if no proper superset is also valid
    maximal_sets = []
    for mask in range(1, size):
        if not valid[mask]:
            continue
        # Check if mask is maximal: no single vertex can be added while staying valid
        is_maximal = True
        for v in range(n):
            if mask & (1 << v):
                continue
            trial = mask | (1 << v)
            if valid[trial]:
                is_maximal = False
                break
        if is_maximal:
            maximal_sets.append(mask)
    
    # Sort maximal sets by size descending for better pruning
    maximal_sets.sort(key=lambda x: bin(x).count('1'), reverse=True)
    
    # Greedy initial solution for upper bound
    def greedy():
        parts = []
        remaining = full_mask
        while remaining:
            # Pick vertex with highest degree in remaining
            best_v = -1
            best_deg = -1
            for v in range(n):
                if remaining & (1 << v):
                    deg = bin(remaining & adj[v]).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
            # Find largest maximal set subset of remaining containing best_v
            best_part = 1 << best_v
            for ms in maximal_sets:
                if ms & (1 << best_v) and (ms & ~remaining) == 0:
                    if bin(ms).count('1') > bin(best_part).count('1'):
                        best_part = ms
            parts.append(best_part)
            remaining ^= best_part
        return parts
    
    initial_parts = greedy()
    best_count = len(initial_parts)
    best_parts = list(initial_parts)
    
    deadline = time.time() + 55.0
    
    # Branch-and-bound set cover search
    # State: (remaining_mask, selected_sets_list, count)
    # Key insight: we need disjoint sets that cover all vertices
    
    # Precompute for each maximal set, the vertices it contains
    set_vertices = [set() for _ in maximal_sets]
    for i, ms in enumerate(maximal_sets):
        for v in range(n):
            if ms & (1 << v):
                set_vertices[i].add(v)
    
    def search(remaining, selected, count):
        nonlocal best_count, best_parts
        
        if time.time() > deadline:
            return
        
        if remaining == 0:
            if count < best_count:
                best_count = count
                best_parts = list(selected)
            return
        
        # Lower bound: at least ceil(popcount(remaining) / max_set_size) more sets needed
        # But simpler: if count + 1 >= best_count, prune (we need at least 1 more set)
        if count >= best_count:
            return
        
        # Pick the vertex with fewest available maximal sets that can cover it
        # (i.e., maximal sets that are subsets of remaining and contain this vertex)
        min_v = -1
        min_options = n + 1
        
        for v in range(n):
            if not (remaining & (1 << v)):
                continue
            options = 0
            for i, ms in enumerate(maximal_sets):
                if (ms & (1 << v)) and (ms & ~remaining) == 0:
                    options += 1
            if options < min_options:
                min_options = options
                min_v = v
        
        if min_v == -1:
            return
        
        # Try each maximal set that contains min_v and is subset of remaining
        for i, ms in enumerate(maximal_sets):
            if not (ms & (1 << min_v)):
                continue
            if ms & ~remaining:
                continue
            
            new_remaining = remaining ^ ms
            new_selected = selected + [ms]
            search(new_remaining, new_selected, count + 1)
    
    search(full_mask, [], 0)
    
    # Reconstruct result
    result = []
    for pm in best_parts:
        part = []
        for v in range(n):
            if pm & (1 << v):
                part.append(v)
        result.append(part)
    
    # Verify
    all_vertices = set()
    for p in result:
        for v in p:
            all_vertices.add(v)
    if all_vertices != set(range(n)) or len(result) == 0:
        # Fallback: single vertex per part
        result = [[v] for v in range(n)]
    
    final_result = [[int(v) for v in part] for part in result]
    return final_result

# EVOLVE-BLOCK-END
