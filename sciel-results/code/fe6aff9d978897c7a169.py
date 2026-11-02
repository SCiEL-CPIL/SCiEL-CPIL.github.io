import numpy as np

def construct():
    n, m, p = 3, 5, 7
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                r = (i * m + k) * p + j
                
                # U row index: i*m + k
                u_idx = i * m + k
                U[u_idx, r] = 1
                
                # V row index: k*p + j
                v_idx = k * p + j
                V[v_idx, r] = 1
                
                # W row index: j*n + i (based on W[k*n+i, r] convention where k is col, i is row)
                w_idx = j * n + i
                W[w_idx, r] = 1
                
    return {'U': U, 'V': V, 'W': W}
