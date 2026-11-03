import numpy as np

def construct():
    m = 3
    n = 3
    p = 6
    rank = m * n * p
    
    U = np.zeros((m * n, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Map r to (i, j, k)
    # r = i * (n * p) + j * p + k
    # i in [0, m), j in [0, n), k in [0, p)
    
    for i in range(m):
        for j in range(n):
            for k in range(p):
                r = i * (n * p) + j * p + k
                
                # U index: row i, col j of A. Flattened: i * n + j
                u_idx = i * n + j
                U[u_idx, r] = 1
                
                # V index: row j, col k of B. Flattened: j * p + k
                v_idx = j * p + k
                V[v_idx, r] = 1
                
                # W index: Output C[i, k]. Flattened as k * n + i per prompt "W[k*n+i,r]"
                w_idx = k * n + i
                W[w_idx, r] = 1
                
    return {'U': U, 'V': V, 'W': W}
