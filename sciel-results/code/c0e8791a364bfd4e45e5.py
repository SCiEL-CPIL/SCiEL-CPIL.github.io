import numpy as np

def construct():
    n, m, p = 2, 4, 7
    # Direct Schoolbook Decomposition
    # Rank = n * m * p = 2 * 4 * 7 = 56
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over all combinations of i (row of A), j (inner index), k (col of B)
    # A is n x m, B is m x p, C is n x p
    # C[i, k] += A[i, j] * B[j, k]
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i  (Trace convention W[k*n+i, r])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
