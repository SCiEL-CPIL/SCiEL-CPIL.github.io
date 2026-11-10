import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_indep for all masks
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest]:
            if (adj_mask[v] & rest) == rest:
                is_clique[mask] = True
            else:
                is_clique[mask] = False
        else:
            is_clique[mask] = False
            
        if is_indep[rest]:
            if (adj_mask[v] & rest) == 0:
                is_indep[mask] = True
            else:
                is_indep[mask] = False
        else:
            is_indep[mask] = False
            
    # Precompute valid: non-empty masks that are clique or independent
    valid = [False] * (1 << n)
    valid[0] = False
    for mask in range(1, 1 << n):
        valid[mask] = is_clique[mask] or is_indep[mask]
        
    # DP: dp[mask] = minimum number of parts to partition mask
    # Initialize with infinity
    INF = n + 1
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    # parent tracking for reconstruction
    parent = [-1] * (1 << n)  # parent[mask] = the submask that was removed
    
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
        # Try all valid submasks of mask
        sub = mask
        while sub:
            if valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
    
    # Reconstruct the partition
    if dp[full_mask] >= INF:
        # Fallback: greedy
        es = {tuple(e) for e in edges}
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts
    
    # Reconstruct
    parts = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        if sub <= 0:
            # Should not happen if dp was computed correctly
            break
        part = [j for j in range(n) if (sub >> j) & 1]
        parts.append(part)
        mask ^= sub
    
    # Convert to plain Python lists of ints
    out = []
    for part in parts:
        out.append([int(x) for x in part])
    
    return out
