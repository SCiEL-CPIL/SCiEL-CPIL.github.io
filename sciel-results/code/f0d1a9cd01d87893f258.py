import numpy as np

def construct():
    n, m, p = 2, 2, 8
    # A is n x m (2x2)
    # B is m x p (2x8)
    # C is n x p (2x8)
    
    # Trivial decomposition:
    # C[i, j] = sum_k A[i, k] B[k, j]
    # Terms: (i, k, j)
    # Index in A: i*m + k
    # Index in B: k*p + j
    # Index in C: i*p + j
    
    rank = n * m * p  # 2 * 2 * 8 = 32
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # U[r, i*m+k] = 1
                U[r, i * m + k] = 1.0
                # V[r, k*p+j] = 1
                V[r, k * p + j] = 1.0
                # W[r, i*p+j] = 1
                W[r, i * p + j] = 1.0
                r += 1
                
    return {'U': U, 'V': V, 'W': W}
