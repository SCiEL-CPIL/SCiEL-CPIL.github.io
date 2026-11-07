import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_masks = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    size = 1 << n
    full_mask = size - 1
    
    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check if mask is a clique:
        # rest must be a clique, and v must be connected to all in rest
        is_clique[mask] = is_clique[rest] and ((adj_masks[v] & rest) == rest)
        
        # Check if mask is an independent set:
        # rest must be an independent set, and v must NOT be connected to any in rest
        is_indep[mask] = is_indep[rest] and ((adj_masks[v] & rest) == 0)

    # DP to find minimum number of parts
    # dp[mask] = min number of homogeneous sets to cover mask
    dp = [float('inf')] * size
    parent = [-1] * size
    dp[0] = 0
    
    # To ensure we finish within time, we can add a check, but 3^14 is ~5M ops, which is fast.
    # We'll iterate masks in increasing order.
    for mask in range(1, size):
        # Iterate over all non-empty submasks of mask
        sub = mask
        while sub:
            if is_clique[sub] or is_indep[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
            
    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask:
        s = parent[mask]
        if s == -1:
            # Fallback: should not happen, but just in case
            # Greedy assignment
            es = {tuple(e) for e in edges}
            parts = []
            for v in range(n):
                for part in parts:
                    trial = part + [v]
                    hits = [(u, w) in es or (w, u) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                    if all(hits) or not any(hits):
                        part.append(v)
                        break
                else:
                    parts.append([v])
            return parts
            
        part = []
        for i in range(n):
            if s & (1 << i):
                part.append(i)
        parts.append(part)
        mask ^= s
        
    return parts
