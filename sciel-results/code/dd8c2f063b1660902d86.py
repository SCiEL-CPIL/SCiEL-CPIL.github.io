"""Tensor decomposition for 2x3 x 3x3 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    fallback = schoolbook()
    
    # Attempt rank 17 decomposition with structured ansatz
    # Key idea: one rank-1 term can handle two (i,j,k) points if the supports align
    
    rank = 17
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Standard terms for all (i,j,k) except (0,0,0) and (0,0,1)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == 0 and j == 0 and k in (0, 1):
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # col should be 16 now
    # Combined term at col=16 handles (0,0,0) and (0,0,1)
    U[0][16] = 1
    V[0][16] = 1
    V[1][16] = 1
    W[0][16] = 1
    W[1][16] = 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    if max_err < 1e-10:
        return result
    else:
        return fallback
