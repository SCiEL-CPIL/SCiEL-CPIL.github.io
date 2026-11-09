import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 118  # 150s limit, leave 32s for safety/overhead
    
    # Fallback: Andrasfai graph And(9) on 26 vertices
    # Triangle-free, alpha = 9
    best_result = {
        'n': 26,
        'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    }
    
    # Try to find a larger graph
    # R(3, 10) <= 41, so max n is 40
    
    # We will try n from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        # Time budget per n: prioritize higher n
        # Give more time to n=40 and n=39
        if n == 40:
            time_budget = 60
        elif n == 39:
            time_budget = 40
        elif n == 38:
            time_budget = 20
        else:
            time_budget = 10
        
        n_start = time.time()
        
        restarts = 0
        max_restarts = 100000  # Limit by time, not count
        
        while time.time() - n_start < time_budget and time.time() < deadline - 5 and restarts < max_restarts:
            restarts += 1
            
            # Generate random maximal triangle-free graph
            # Strategy: Random ordering of edges, add if no triangle formed
            adj = [0] * n
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check if alpha < 10
            # Precompute non_adj
            all_mask = (1 << n) - 1
            non_adj = [0] * n
            for i in range(n):
                non_adj[i] = (all_mask ^ (1 << i)) & ~adj[i]
            
            # Fast independent set checker
            # We want to verify if there is NO independent set of size 10.
            # Equivalent to: Maximum Independent Set < 10.
            
            def has_independent_set_10():
                # DFS with pruning
                # State: mask of available vertices, size of current independent set
                # Heuristic: Pick vertex with max degree in induced subgraph?
                # Or min degree?
                # For independent set, if we pick v, we remove its neighbors.
                # If we have many neighbors, we remove many vertices, shrinking the search space faster.
                # So max degree is usually a good heuristic for pruning.
                
                # Optimization: Precompute degrees for all vertices in the full graph?
                # No, degrees change as mask changes.
                
                # Let's use a simple recursive search with memoization on mask?
                # Mask space is 2^n, which is too large for n=40.
                # But depth is only 10.
                
                def dfs(mask, size):
                    if size == 10:
                        return True
                    if mask == 0:
                        return False
                    
                    # Pruning: if remaining vertices + size < 10, fail
                    if mask.bit_count() < 10 - size:
                        return False
                    
                    # Heuristic: Pick vertex with max degree in mask
                    max_deg = -1
                    best_v = -1
                    temp = mask
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        deg = (adj[v] & mask).bit_count()
                        if deg > max_deg:
                            max_deg = deg
                            best_v = v
                        temp ^= lsb
                        
                    if best_v == -1:
                        return False
                    
                    v_bit = 1 << best_v
                    
                    # Branch 1: Include best_v
                    # New mask: non-neighbors of best_v in mask
                    new_mask_include = non_adj[best_v] & mask
                    if dfs(new_mask_include, size + 1):
                        return True
                    
                    # Branch 2: Exclude best_v
                    new_mask_exclude = mask & ~v_bit
                    if dfs(new_mask_exclude, size):
                        return True
                    
                    return False
                
                return dfs(all_mask, 0)
            
            if not has_independent_set_10():
                # Valid graph found
                edges = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask ^= lsb
                return {'n': n, 'edges': edges}
            
            # Optional: Local Search / Edge Removal Strategy?
            # If we find an independent set of size 10, we could try to add edges to break it?
            # But adding edges might create triangles.
            # The current random generation produces maximal triangle-free graphs.
            # If alpha >= 10, we can't easily fix it by adding edges (already maximal).
            # We could try to *remove* edges from the complement (i.e., add edges to G) but we can't.
            # We could try to *remove* edges from G? No, that increases alpha.
            
            # So we just restart.
            
    return best_result
