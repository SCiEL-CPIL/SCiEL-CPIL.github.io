import time
import sys
import random

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    
    # The trivial decomposition of the matrix multiplication tensor M_{n,m,p}
    # has rank m.
    # C_{i,k} = sum_{j=0}^{m-1} A_{i,j} B_{j,k}
    # This corresponds to a sum of m rank-1 tensors.
    # Rank r corresponds to the summation index j = r.
    
    rank = m
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U
    # U index: i*m + j
    # For rank r (which is j), U[i*m + r, r] = 1 for all i
    for i in range(n):
        for r in range(m):
            u_idx = i * m + r
            U[u_idx][r] = 1
            
    # Fill V
    # V index: j*p + k
    # For rank r (which is j), V[r*p + k, r] = 1 for all k
    for r in range(m):
        for k in range(p):
            v_idx = r * p + k
            V[v_idx][r] = 1
            
    # Fill W
    # W index: k*n + i
    # For rank r, W[k*n + i, r] = 1 for all k, i
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            for r in range(m):
                W[w_idx][r] = 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
