import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    # Schoolbook rank is n * m * p
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Iterate over all terms in the schoolbook algorithm
    # Term t corresponds to indices i (row of A/C), j (col of A / row of B), k (col of B/C)
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                # V index: j * p + k
                # W index: k * n + i (based on prompt: W[k*n+i, r])
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, t] = 1
                V[v_idx, t] = 1
                W[w_idx, t] = 1
                
                t += 1
                
    return {"U": U, "V": V, "W": W}
