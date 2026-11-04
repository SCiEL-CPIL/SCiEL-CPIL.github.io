import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to reduce rank using Strassen on 2x2 sub-blocks
    # For 2x3 * 3x10, we can treat the 3x10 as two 3x5 blocks or use other decompositions
    # Let's try a different approach: use the fact that we can combine some terms
    
    # Actually, let's try a more systematic approach using tensor decomposition
    # The tensor T[i,j,k] = e_i e_j e_k for matrix multiplication
    # We want to find rank r such that T = sum_{t=1}^r u_t v_t w_t
    
    # For now, return the schoolbook solution as baseline
    return {"U": U, "V": V, "W": W}
