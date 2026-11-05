import numpy as np

def construct():
    n, m, p = 2, 4, 7
    r = m  # Rank 4
    
    # U shape (n*m, r)
    # U[i*m + j, c] = delta_{j,c}
    U = np.zeros((n * m, r))
    for i in range(n):
        for j in range(m):
            for c in range(r):
                if j == c:
                    U[i * m + j, c] = 1.0
                    
    # V shape (m*p, r)
    # V[j*p + k, c] = (D_k^{-1})_{c,j}
    # We can choose D_k = I_m for all k.
    # Then V[j*p + k, c] = delta_{c,j}
    V = np.zeros((m * p, r))
    for j in range(m):
        for k in range(p):
            for c in range(r):
                if c == j:
                    V[j * p + k, c] = 1.0
                    
    # W shape (p*n, r)
    # W[k*n + i, c] = (D_k)_{c,c} = 1
    W = np.zeros((p * n, r))
    for k in range(p):
        for i in range(n):
            for c in range(r):
                W[k * n + i, c] = 1.0
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
