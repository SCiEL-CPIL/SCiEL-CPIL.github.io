import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Precompute popcount for 17-bit integers? Or just use bin().count('1')
    # bin().count('1') is fast enough for 17 bits.
    
    best_subset = list(range(min_vertices))
    best_distinct = 0
    start_time = time.time()
    
    # DFS with pruning
    # State: idx (next vertex to consider), included_mask, deg_list (degrees of included vertices)
    # We need to map included_mask to actual vertex indices to update deg_list correctly.
    # It's easier to keep a list of included vertices and their degrees.
    
    def dfs(idx, included_mask, included_vertices, deg_list, distinct_count):
        nonlocal best_subset, best_distinct
        
        # Deadline check
        if time.time() - start_time > 85:
            return
            
        # Pruning: Can we reach min_vertices?
        remaining = n - idx
        if len(included_vertices) + remaining < min_vertices:
            return
            
        # Pruning: Can we beat best_distinct?
        # Max distinct degrees in a graph of size K is K.
        # Current size is len(included_vertices).
        # Max possible size is len(included_vertices) + remaining.
        # So max possible distinct degrees is len(included_vertices) + remaining.
        if len(included_vertices) + remaining < best_distinct:
            return
            
        # If we've considered all vertices
        if idx == n:
            if len(included_vertices) >= min_vertices:
                # distinct_count is already maintained, but let's verify or just use it
                # Actually, distinct_count is passed in, but we need to ensure it's correct.
                # It is maintained in the recursion.
                if distinct_count > best_distinct:
                    best_distinct = distinct_count
                    best_subset = list(included_vertices)
            return
            
        # Option 1: Exclude vertex idx
        dfs(idx + 1, included_mask, included_vertices, deg_list, distinct_count)
        
        # Option 2: Include vertex idx
        # Check if including it is valid (always valid, just updates state)
        new_included_mask = included_mask | (1 << idx)
        
        # Calculate degree of idx in the new subgraph
        # Neighbors of idx that are in included_mask
        # Note: included_mask does NOT contain idx yet.
        deg_idx = bin(adj_mask[idx] & included_mask).count('1')
        
        # Update degrees of existing neighbors
        # We need to know which vertices in included_vertices are neighbors of idx.
        # We can iterate through included_vertices.
        # Since n is small, this is fine.
        new_deg_list = list(deg_list)
        for v in included_vertices:
            if adj_mask[idx] & (1 << v):
                # v is a neighbor of idx
                # Find v's index in new_deg_list?
                # This is O(N^2) if we search.
                # Better: store degrees in a dict or parallel list.
                # Let's use a dict {vertex: degree} for clarity, though list is faster if indexed by vertex.
                pass
        
        # Let's refactor to use a list `deg` of length n, where deg[v] is degree if v is included, else -1.
        # This makes updates O(1) per neighbor.
        
        # Re-implementing DFS with `deg` array
        pass

    # Let's restart the DFS implementation with a better state representation.
    
    deg_array = [-1] * n
    included_vertices = []
    
    def dfs2(idx, included_mask, distinct_count):
        nonlocal best_subset, best_distinct, deg_array, included_vertices
        
        if time.time() - start_time > 85:
            return
            
        remaining = n - idx
        cur_size = len(included_vertices)
        
        if cur_size + remaining < min_vertices:
            return
        if cur_size + remaining < best_distinct:
            return
            
        if idx == n:
            if cur_size >= min_vertices and distinct_count > best_distinct:
                best_distinct = distinct_count
                best_subset = list(included_vertices)
            return
            
        # Exclude idx
        dfs2(idx + 1, included_mask, distinct_count)
        
        # Include idx
        # 1. Calculate new degree of idx
        deg_idx = bin(adj_mask[idx] & included_mask).count('1')
        
        # 2. Update degrees of neighbors
        old_degrees_to_restore = []
        new_distinct_set = set()
        
        # To update distinct_count efficiently, we should maintain the set of degrees.
        # But updating a set with increments is tricky.
        # Given N=17, we can just recompute distinct_count at the end of the branch?
        # No, that's O(N) per node. 2^17 * 17 = 2M ops. Fine.
        
        # Let's just recompute distinct_count from deg_array after updates.
        # This simplifies state management.
        
        # Save old degrees of neighbors
        neighbors_in_mask = adj_mask[idx] & included_mask
        # Iterate through set bits in neighbors_in_mask
        temp_mask = neighbors_in_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            old_degrees_to_restore.append((v, deg_array[v]))
            deg_array[v] += 1
            temp_mask ^= lsb
            
        # Add idx
        deg_array[idx] = deg_idx
        included_vertices.append(idx)
        
        # Recompute distinct count
        # Only for included vertices
        d_set = set()
        for v in included_vertices:
            d_set.add(deg_array[v])
        new_distinct_count = len(d_set)
        
        dfs2(idx + 1, included_mask | (1 << idx), new_distinct_count)
        
        # Backtrack
        included_vertices.pop()
        deg_array[idx] = -1
        for v, old_deg in old_degrees_to_restore:
            deg_array[v] = old_deg
            
    dfs2(0, 0, 0)
    
    return [int(v) for v in best_subset]

construct()
