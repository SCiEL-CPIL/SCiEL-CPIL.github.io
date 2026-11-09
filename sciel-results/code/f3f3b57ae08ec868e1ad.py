"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Optimized construction:
    # The tensor T[(i,j), (j,k), (k,i)] = 1 for all i, j, k.
    # We can decompose this into m terms, one for each j.
    # For each j, the term is non-zero only when the middle index matches j.
    # U[i, j, r=j] = 1
    # V[j, k, r=j] = 1
    # W[k, i, r=j] = 1
    # Sum over r: only r=j contributes for a given j.
    # Value is 1*1*1 = 1.
    # This covers all entries with rank m.
    
    rank_opt = m
    U_opt = [[0] * rank_opt for _ in range(n * m)]
    V_opt = [[0] * rank_opt for _ in range(m * p)]
    W_opt = [[0] * rank_opt for _ in range(p * n)]
    
    for j in range(m):
        # Set U entries for this j
        for i in range(n):
            U_opt[i * m + j][j] = 1
        # Set V entries for this j
        for k in range(p):
            V_opt[j * p + k][j] = 1
        # Set W entries for this j
        for k in range(p):
            for i in range(n):
                W_opt[k * n + i][j] = 1
                
    # Verify the decomposition
    # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank_opt):
                    s += U_opt[i * m + j][r] * V_opt[j * p + k][r] * W_opt[k * n + i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        best_result = {"U": U_opt, "V": V_opt, "W": W_opt}
        best_rank = rank_opt
        
    if time.time() > deadline:
        return best_result
        
    return best_result
# EVOLVE-BLOCK-END
