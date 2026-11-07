"""Optimized search with greedy local refinement for distinct degrees."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def popcount(x):
        return bin(x).count('1')
    
    def get_degrees(subset_mask):
        """Return dict of vertex -> degree for vertices in subset."""
        degs = {}
        for v in range(n):
            if subset_mask & (1 << v):
                degs[v] = popcount(adj_mask[v] & subset_mask)
        return degs
    
    def count_distinct_degrees(subset_mask):
        """Count distinct degrees in the induced subgraph of subset_mask."""
        degs = get_degrees(subset_mask)
        return len(set(degs.values()))
    
    def local_search(subset_mask, current_best):
        """Try to improve by swapping vertices in/out. Returns (best_mask, best_count)."""
        best_mask = subset_mask
        best_count = current_best
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            in_vertices = [v for v in range(n) if subset_mask & (1 << v)]
            out_vertices = [v for v in range(n) if not (subset_mask & (1 << v))]
            
            # Try adding one vertex
            for add_v in out_vertices:
                new_mask = subset_mask | (1 << add_v)
                new_count = count_distinct_degrees(new_mask)
                if new_count > best_count:
                    best_count = new_count
                    best_mask = new_mask
                    subset_mask = new_mask
                    improved = True
            
            # Try removing one vertex (only if size > min_v)
            if popcount(subset_mask) > min_v:
                for rem_v in in_vertices:
                    new_mask = subset_mask & ~(1 << rem_v)
                    new_count = count_distinct_degrees(new_mask)
                    if new_count > best_count:
                        best_count = new_count
                        best_mask = new_mask
                        subset_mask = new_mask
                        improved = True
            
            # Try swapping one vertex
            for add_v in out_vertices:
                for rem_v in in_vertices:
                    if time.time() > deadline:
                        return best_mask, best_count
                    new_mask = (subset_mask & ~(1 << rem_v)) | (1 << add_v)
                    new_count = count_distinct_degrees(new_mask)
                    if new_count > best_count:
                        best_count = new_count
                        best_mask = new_mask
                        subset_mask = new_mask
                        improved = True
        
        return best_mask, best_count
    
    best_count = 0
    best_mask = 0
    
    # Phase 1: Exhaustive search for size min_v
    for subset in itertools.combinations(range(n), min_v):
        if time.time() > deadline:
            break
        mask = 0
        for v in subset:
            mask |= (1 << v)
        distinct = count_distinct_degrees(mask)
        if distinct > best_count:
            best_count = distinct
            best_mask = mask
    
    # Phase 2: Local search from best found
    if best_mask != 0:
        best_mask, best_count = local_search(best_mask, best_count)
    
    # Phase 3: Try other sizes with exhaustive + local search
    for size in range(min_v + 1, n + 1):
        if time.time() > deadline:
            break
        
        size_best_mask = 0
        size_best_count = 0
        
        for subset in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break
            mask = 0
            for v in subset:
                mask |= (1 << v)
            distinct = count_distinct_degrees(mask)
            if distinct > size_best_count:
                size_best_count = distinct
                size_best_mask = mask
        
        if size_best_mask != 0:
            # Local search from this size's best
            lm, lc = local_search(size_best_mask, size_best_count)
            if lc > best_count:
                best_count = lc
                best_mask = lm
    
    # Convert best_mask to list of vertices
    if best_mask == 0:
        best_subset = list(range(min_v))
    else:
        best_subset = [v for v in range(n) if best_mask & (1 << v)]
    
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
