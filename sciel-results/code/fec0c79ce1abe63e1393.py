import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    deadline = time.time() + 130.0
    n, m, p = DIMENSIONS  # n=3, m=5, p=6
    
    # Schoolbook decomposition: rank = n*m*p = 90
    # U: (n*m, rank) - extracts A[i,j]
    # V: (m*p, rank) - extracts B[j,k]
    # W: (p*n, rank) - extracts C[k*n+i] per trace convention W[k*n+i, r]
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] is at index i*m + j in U
                # B[j,k] is at index j*p + k in V
                # C[i,k] is at index k*n + i in W (trace convention)
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
