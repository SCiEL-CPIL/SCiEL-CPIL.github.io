import numpy as np

def construct():
    n, m, p = 2, 3, 3
    rank = 17
    
    # U shape: (n*m, rank) = (6, 17)
    # V shape: (m*p, rank) = (9, 17)  
    # W shape: (p*n, rank) = (6, 17)
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 for all i,j,k
    
    # 10 schoolbook terms for (j=2 or k=2)
    c = 0
    for i in range(n):
        for j in [2]:  # j=2, k=0,1,2
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
        for k in [2]:  # j=0,1, k=2
            for j in range(m-1):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    
    # 7 Strassen terms for the 2x2x2 subproblem (i in {0,1}, j in {0,1}, k in {0,1})
    # Need to find valid U, V, W for these 7 terms
    
    # Use known Strassen decomposition
    strassen_U = np.array([
        [1, 0, 1, 0, 1, -1, 1],
        [1, 1, 0, 1, 1,  1, -1]
    ])
    strassen_V = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1]
    ])
    strassen_W = np.array([
        [1, 1, 1, -1, 0, 1, 1],
        [1, 0, -1, 1, 1, 0, 1]
    ])
    
    for cc in range(7):
        for i in range(2):
            U[i*m + [0,1][i], c + cc] = strassen_U[i, cc]
        for j in range(2):
            V[j*p + [0,1][j], c + cc] = strassen_V[j, cc]
        for k in range(2):
            W[k*n + [0,1][k], c + cc] = strassen_W[k, cc]
    
    # Verify
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    assert np.allclose(T, 1), f"Verification failed: {T}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
