import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Naive decomposition
    # C_ij = sum_k A_ik B_kj
    # Rank = n * m * p = 4 * 4 * 8 = 128
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for k in range(m): # inner index
        for i in range(n): # row of A/C
            for j in range(p): # col of B/C
                t = k * n * p + i * p + j
                # U index: A is n x m. A_ik is at i*m + k
                U[i * m + k, t] = 1.0
                # V index: B is m x p. B_kj is at k*p + j
                V[k * p + j, t] = 1.0
                # W index: C is n x p. C_ij is at j*n + i (per problem statement W[k*n+i,r] where k is col, i is row)
                W[j * n + i, t] = 1.0
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
