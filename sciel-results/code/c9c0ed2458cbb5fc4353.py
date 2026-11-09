import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s for safety/overhead
    
    # Fallback: Andrasfai graph And(9) on 26 vertices
    # Triangle-free, alpha = 9
    best_result = {
        'n': 26,
        'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    }
    
    # We know R(3,10) > 39, so a graph on 39 vertices exists.
    # R(3,10) <= 41, so n <= 40.
    # We try n=40, then n=39, etc.
    
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        restarts = 0
        max_restarts = 20000 # High restart count, but filtered by fast greedy
        
        while time.time() < deadline - 2 and restarts < max_restarts:
            restarts += 1
            
            # Generate random maximal triangle-free graph
            adj = [0] * n
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Precompute masks
            all_mask = (1 << n) - 1
            non_adj = [0] * n
            for i in range(n):
                non_adj[i] = (all_mask ^ (1 << i)) & ~adj[i]
            
            # Stage 1: Greedy check for independent set of size 10
            # If greedy finds one, alpha >= 10, so graph is INVALID for our purpose.
            # We want alpha < 10.
            if greedy_has_independent_set_10(adj, n, all_mask, non_adj):
                continue
            
            # Stage 2: Exact check
            # If greedy didn't find one, it might still exist.
            # Use DFS to be sure.
            if not exact_has_independent_set_10(adj, n, all_mask, non_adj):
                # Valid graph found: Triangle-free (by construction) and alpha < 10
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
        
    return best_result

def greedy_has_independent_set_10(adj, n, all_mask, non_adj):
    """
    Returns True if a greedy algorithm finds an independent set of size 10.
    This is a necessary condition for alpha >= 10, but not sufficient.
    If it returns False, we don't know for sure, so we need exact check.
    If it returns True, we know alpha >= 10.
    """
    mask = all_mask
    size = 0
    while mask:
        if size >= 10:
            return True
        
        # Pick vertex with minimum degree in the induced subgraph
        min_deg = n + 1
        best_v = -1
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v == -1:
            break
            
        size += 1
        # Remove best_v and its neighbors
        mask &= non_adj[best_v]
        
    return size >= 10

def exact_has_independent_set_10(adj, n, all_mask, non_adj):
    """
    Returns True if there exists an independent set of size 10.
    Uses branch and bound.
    """
    def dfs(mask, size):
        if size == 10:
            return True
        if mask == 0:
            return False
        if mask.bit_count() + size < 10:
            return False
        
        # Heuristic: pick vertex with maximum degree in induced subgraph
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
        
        # Include best_v
        if dfs(non_adj[best_v] & mask, size + 1):
            return True
        
        # Exclude best_v
        if dfs(mask & ~v_bit, size):
            return True
        
        return False
    
    return dfs(all_mask, 0)
