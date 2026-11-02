import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 12
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Term r corresponds to (i, j, k)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][r] = 1
                V[j*p + k][r] = 1
                W[k*n + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
