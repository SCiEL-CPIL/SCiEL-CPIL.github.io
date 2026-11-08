import numpy as np

def construct():
    n, m, p = 2, 4, 6
    U_rows = n * m  # 8
    V_rows = m * p  # 24
    W_rows = p * n  # 12
    
    # The tensor T[i,j,k] is 1 if j==k (and j < m, k < p), else 0.
    # It is independent of i.
    # We can decompose this using the rank of the m x p diagonal submatrix.
    # Since the non-zero part is the m x m identity matrix (embedded in m x p),
    # its rank is m.
    rank = m  # 4
    
    U = [[0.0] * rank for _ in range(U_rows)]
    V = [[0.0] * rank for _ in range(V_rows)]
    W = [[0.0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        # U[i*m + j, c] = 1 if j == c
        for i in range(n):
            U[i * m + c][c] = 1.0
            
        # V[j*p + k, c] = 1 if k == c
        for j in range(m):
            V[j * p + c][c] = 1.0
            
        # W[k*n + i, c] = 1 for all i, k
        for k in range(p):
            for i in range(n):
                W[k * n + i][c] = 1.0
                
    return {"U": U, "V": V, "W": W}
