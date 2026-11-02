"""Program for matrix multiplication 2x2 by 2x12 tensor decomposition."""
import time

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # Trivial schoolbook decomposition with rank = n*m*p = 2*2*12 = 48
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][idx] = 1
                # V index: j*p + k
                V[j * p + k][idx] = 1
                # W index: k*n + i (official trace convention)
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}
