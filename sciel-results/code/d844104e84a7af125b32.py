import numpy as np

def construct():
    m, n, p = 3, 4, 8
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for j in range(n):
        for i in range(m):
            for k in range(p):
                # U index: column-major for A (m x n) -> j * m + i
                u_idx = j * m + i
                # W index: column-major for B (n x p) -> k * n + j
                w_idx = k * n + j
                # V index: column-major for C (m x p) -> k * m + i
                v_idx = k * m + i
                
                U[u_idx, r] = 1
                W[w_idx, r] = 1
                V[v_idx, r] = 1
                
                r += 1
                
    return {'U': U, 'V': V, 'W': W}
