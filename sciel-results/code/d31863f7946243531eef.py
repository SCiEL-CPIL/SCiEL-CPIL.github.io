# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    R = 4
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.ones((p * n, R))
    
    for i in range(n):
        for j in range(m):
            U[i * m + j, j] = 1.0
            
    for j in range(m):
        for k in range(p):
            V[j * p + k, j] = 1.0
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
