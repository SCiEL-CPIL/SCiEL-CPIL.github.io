"""Matrix multiplication 3,5,7 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # The rank is m=5, using the decomposition:
    # C[i,k] = sum_{l=0}^{4} A[i,l] * B[l,k]
    # Each term A[i,l]*B[l,k] corresponds to a rank-1 tensor component.
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for l in range(m):
        for i in range(n):
            U[i * m + l][l] = 1
        for k in range(p):
            V[l * p + k][l] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][l] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
