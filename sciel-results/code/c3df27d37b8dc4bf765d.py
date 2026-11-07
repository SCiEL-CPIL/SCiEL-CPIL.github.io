import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute valid masks (clique or independent set)
    valid = [False] * (1 << n)
    valid[0] = True
    
    for mask in range(1, 1 << n):
        # Check clique
        is_c = True
        m = mask
        while m and is_c:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if (adj[i] & mask) != (mask ^ lsb):
                is_c = False
            m ^= lsb
        
        if is_c:
            valid[mask] = True
            continue
        
        # Check independent set
        is_i = True
        m = mask
        while m and is_i:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if (adj[i] & mask) != 0:
                is_i = False
            m ^= lsb
        
        if is_i:
            valid[mask] = True
    
    # Subset DP for minimum cochromatic partition
    INF = n + 1
    dp = [INF] * (1 << n)
    dp[0] = 0
    parent = [-1] * (1 << n)
    
    for mask in range(1, 1 << n):
        sub = mask
        while sub > 0:
            if valid[sub]:
                rest = mask ^ sub
                if dp[rest] + 1 < dp[mask]:
                    dp[mask] = dp[rest] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
    
    # Reconstruct partition
    partition = []
    mask = full_mask
    while mask > 0:
        sub = parent[mask]
        part = [i for i in range(n) if sub & (1 << i)]
        partition.append(part)
        mask ^= sub
    
    # Ensure all types are correct: list of lists of int
    result = []
    for part in partition:
        result.append([int(v) for v in part])
    
    return result

# EVOLVE-BLOCK-END
