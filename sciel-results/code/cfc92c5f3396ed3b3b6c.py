import numpy as np

def construct():
    n = 3
    m = 3
    p = 5
    
    # Naive rank
    rank = n * m * p
    
    # Initialize matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Fill the decomposition
    # The tensor T[i,j,k] = 1
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We assign a unique r for each (i,j,k)
    r_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Indices for U
                u_idx = i * m + j
                # Indices for V
                v_idx = j * p + k
                # Indices for W
                w_idx = k * n + i
                
                U[u_idx, r_idx] = 1.0
                V[v_idx, r_idx] = 1.0
                W[w_idx, r_idx] = 1.0
                
                r_idx += 1
                
    return {'U': U, 'V': V, 'W': W}
