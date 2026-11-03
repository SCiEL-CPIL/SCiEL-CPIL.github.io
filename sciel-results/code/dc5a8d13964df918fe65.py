import numpy as np

def construct():
    n, m, p = 3, 5, 7
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Index mapping:
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U extracts A[i, j]
                u_idx = i * m + j
                U[u_idx, r] = 1.0
                
                # V extracts B[j, k]
                v_idx = j * p + k
                V[v_idx, r] = 1.0
                
                # W places result in C[i, k]
                # W index is k * n + i
                w_idx = k * n + i
                W[w_idx, r] = 1.0
                
                r += 1
                
    return {'U': U, 'V': V, 'W': W}
