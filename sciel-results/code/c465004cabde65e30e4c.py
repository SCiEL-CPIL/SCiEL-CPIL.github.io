"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill
    
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
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        if is_independent[rest] and (adj[v] & rest) == 0:
            is_independent[mask] = True
            
    valid = [is_clique[i] or is_independent[i] for i in range(size)]
    
    # Greedy upper bound for pruning
    def greedy_upper_bound():
        remaining = (1 << n) - 1
        count = 0
        while remaining:
            # Find largest valid subset of remaining
            best = 0
            best_size = 0
            sub = remaining
            while sub:
                if valid[sub]:
                    sz = bin(sub).count('1')
                    if sz > best_size:
                        best_size = sz
                        best = sub
                sub = (sub - 1) & remaining
            if best:
                remaining ^= best
                count += 1
            else:
                # Should not happen
                remaining ^= (remaining & -remaining)
                count += 1
        return count
    
    ub = greedy_upper_bound()
    
    # DP
    dp = [10**9] * size
    parent = [-1] * size
    dp[0] = 0
    
    for mask in range(1, size):
        if time.time() > deadline:
            # Fallback: use greedy result
            remaining = (1 << n) - 1
            parts = []
            while remaining:
                best = 0
                best_size = 0
                sub = remaining
                while sub:
                    if valid[sub]:
                        sz = bin(sub).count('1')
                        if sz > best_size:
                            best_size = sz
                            best = sub
                    sub = (sub - 1) & remaining
                if best:
                    verts = []
                    temp = best
                    while temp:
                        lsb = temp & -temp
                        verts.append(lsb.bit_length() - 1)
                        temp ^= lsb
                    parts.append(verts)
                    remaining ^= best
                else:
                    lsb = remaining & -remaining
                    parts.append([lsb.bit_length() - 1])
                    remaining ^= lsb
            return parts
            
        # If we already have a solution with 1 part, skip
        if valid[mask]:
            dp[mask] = 1
            parent[mask] = mask
            continue
            
        # Iterate submasks, but only proper submasks (skip mask itself since we checked)
        sub = (mask - 1) & mask
        best_val = 10**9
        best_sub = -1
        while sub:
            if valid[sub]:
                prev = mask ^ sub
                val = dp[prev] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
                    if best_val == 2:
                        break  # Can't do better than 2 for non-valid masks
            sub = (sub - 1) & mask
        
        if best_sub != -1:
            dp[mask] = best_val
            parent[mask] = best_sub
        # else dp[mask] stays 10**9 (shouldn't happen)
            
    # Reconstruct
    full_mask = size - 1
    if dp[full_mask] == 10**9:
        # Fallback: singletons
        return [[i] for i in range(n)]
        
    parts = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        if sub == -1:
            # Shouldn't happen, but handle gracefully
            lsb = mask & -mask
            parts.append([lsb.bit_length() - 1])
            mask ^= lsb
        else:
            verts = []
            temp = sub
            while temp:
                lsb = temp & -temp
                verts.append(lsb.bit_length() - 1)
                temp ^= lsb
            parts.append(verts)
            mask ^= sub
        
    # Ensure types are correct
    return parts
# EVOLVE-BLOCK-END
