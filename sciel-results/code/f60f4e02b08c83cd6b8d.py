import sys
import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    size = 1 << n
    full_mask = size - 1
    
    # Precompute is_clique and is_indep arrays
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (rest & ~adj_mask[v]) == 0:
            is_clique[mask] = True
            
        if is_indep[rest] and (rest & adj_mask[v]) == 0:
            is_indep[mask] = True
            
    # DP
    dp = [float('inf')] * size
    choice = [-1] * size
    dp[0] = 0
    
    for mask in range(1, size):
        lsb = mask & -mask
        pivot_bit = lsb
        rest_bits = mask ^ pivot_bit
        
        best_val = float('inf')
        best_sub = -1
        
        s = rest_bits
        while True:
            sub = s | pivot_bit
            if is_clique[sub] or is_indep[sub]:
                new_mask = mask ^ sub
                val = 1 + dp[new_mask]
                if val < best_val:
                    best_val = val
                    best_sub = sub
            
            if s == 0:
                break
            s = (s - 1) & rest_bits
        
        dp[mask] = best_val
        choice[mask] = best_sub
        
    # Reconstruct
    partition = []
    mask = full_mask
    while mask:
        sub = choice[mask]
        part = []
        m = sub
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part.append(v)
            m -= lsb
        partition.append(part)
        mask ^= sub
        
    return partition

# EVOLVE-BLOCK-END
