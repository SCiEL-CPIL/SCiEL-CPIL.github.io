"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute is_clique and is_independent
    size = 1 << n
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    adj_list = adj
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (adj_list[v] & rest) == rest:
            is_clique[mask] = True
            
        if is_independent[rest] and (adj_list[v] & rest) == 0:
            is_independent[mask] = True
            
    # Precompute valid masks
    valid = [False] * size
    for i in range(1, size):
        if is_clique[i] or is_independent[i]:
            valid[i] = True

    full_mask = size - 1
    
    # Try k = 1, 2, 3, ... using memoized search
    for k in range(1, n + 1):
        if time.time() > deadline:
            return [[i] for i in range(n)]
        
        memo = {}
        
        def can_partition(mask, remaining):
            if remaining == 0:
                return mask == 0
            if mask == 0:
                return True
            key = (mask, remaining)
            if key in memo:
                return memo[key]
            
            sub = mask
            while sub:
                if valid[sub]:
                    if can_partition(mask ^ sub, remaining - 1):
                        memo[key] = True
                        return True
                sub = (sub - 1) & mask
            
            memo[key] = False
            return False
        
        if can_partition(full_mask, k):
            # Reconstruct using DP
            INF = 10**9
            dp = [INF] * size
            parent = [-1] * size
            dp[0] = 0
            
            for m in range(1, size):
                if valid[m]:
                    dp[m] = 1
                    parent[m] = m
                    continue
                s = m
                while s:
                    if valid[s]:
                        prev = m ^ s
                        if dp[prev] + 1 < dp[m]:
                            dp[m] = dp[prev] + 1
                            parent[m] = s
                    s = (s - 1) & m
                    
            parts = []
            mask = full_mask
            while mask:
                s = parent[mask]
                verts = []
                t = s
                while t:
                    lsb = t & -t
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    t ^= lsb
                parts.append(verts)
                mask ^= s
            return parts
    
    return [[i] for i in range(n)]
# EVOLVE-BLOCK-END
