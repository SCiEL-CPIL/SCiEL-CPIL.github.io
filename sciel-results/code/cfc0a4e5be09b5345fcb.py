"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    P = 27720
    MAX_L = 144
    N = 12
    
    # Precompute masks
    values = np.arange(P)[:, None] + np.arange(MAX_L)[None, :] + 1
    masks = np.zeros((P, MAX_L), dtype=np.int32)
    for i in range(1, N + 1):
        masks[values % i == 0] |= (1 << (i - 1))
    
    f_table = np.zeros(P, dtype=np.int32)
    
    for r in range(P):
        if time.time() - start_time > DEADLINE:
            break
        
        # Precompute adj for all L up to MAX_L
        # adj_at_L[i] for L = 1, 2, ..., MAX_L
        # This is 144 * 13 integers, very small
        adj_at_L = [[0] * (N + 1) for _ in range(MAX_L + 1)]
        for L in range(1, MAX_L + 1):
            for i in range(1, N + 1):
                adj_at_L[L][i] = adj_at_L[L-1][i]
            mask = masks[r, L-1]
            m = mask
            while m:
                bit = m & -m
                i = bit.bit_length()
                adj_at_L[L][i] |= (1 << (L-1))
                m -= bit
        
        lo, hi = 1, MAX_L
        while lo < hi:
            mid = (lo + hi) // 2
            adj = adj_at_L[mid]
            
            match_to_pos = [-1] * mid
            success = True
            
            for i in range(1, N + 1):
                visited = [False] * mid
                
                def dfs(i, visited):
                    bits = adj[i]
                    while bits:
                        j = (bits & -bits).bit_length() - 1
                        bits &= bits - 1
                        if not visited[j]:
                            visited[j] = True
                            if match_to_pos[j] == -1 or dfs(match_to_pos[j], visited):
                                match_to_pos[j] = i
                                return True
                    return False
                
                if not dfs(i, visited):
                    success = False
                    break
            
            if success:
                hi = mid
            else:
                lo = mid + 1
        
        f_table[r] = lo
    
    best_r = int(np.argmax(f_table))
    return int(best_r)
# EVOLVE-BLOCK-END
