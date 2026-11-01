# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Rank is n * p
    r = n * p
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Iterate over rank indices c
    # c corresponds to (i0, k0)
    for i0 in range(n):
        for k0 in range(p):
            c = i0 * p + k0
            
            # Fill U
            # U row index is i * m + j
            for i in range(n):
                for j in range(m):
                    if i == i0:
                        U[i * m + j][c] = 1
                    else:
                        U[i * m + j][c] = 0
                        
            # Fill V
            # V row index is j * p + k
            for j in range(m):
                for k in range(p):
                    if k == k0:
                        V[j * p + k][c] = 1
                    else:
                        V[j * p + k][c] = 0
                        
            # Fill W
            # W row index is k * n + i
            for k in range(p):
                for i in range(n):
                    W[k * n + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
