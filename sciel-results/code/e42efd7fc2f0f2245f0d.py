import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Bitmask representation
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute valid masks: a mask is valid if it's a clique or independent set
    # For n=14, there are 2^14 = 16384 masks, so we can precompute all of them
    valid = [False] * (1 << n)
    valid[0] = True  # Empty set is trivially valid
    
    for mask in range(1, 1 << n):
        # Check if mask is a clique
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
        
        # Check if mask is an independent set
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
    
    # Now use subset DP to find minimum partition
    # dp[mask] = minimum number of parts to partition the set 'mask'
    INF = n + 1
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    # parent[mask] = the subset that was taken as one part when solving dp[mask]
    # This helps us reconstruct the partition
    parent = [-1] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Try all subsets of mask that are valid
        # To avoid redundant work, we can iterate submasks
        sub = mask
        while sub > 0:
            if valid[sub]:
                rest = mask ^ sub
                if dp[rest] + 1 < dp[mask]:
                    dp[mask] = dp[rest] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
    
    # Reconstruct the partition
    partition = []
    mask = full_mask
    while mask > 0:
        sub = parent[mask]
        # Convert sub to list of vertices
        part = [i for i in range(n) if sub & (1 << i)]
        partition.append(part)
        mask ^= sub
    
    return partition

# EVOLVE-BLOCK-END
