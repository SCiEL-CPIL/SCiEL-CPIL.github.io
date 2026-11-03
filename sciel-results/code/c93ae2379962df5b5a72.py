DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the trivial decomposition
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: (i, j) -> i*m + j
                U[i * m + j][r] = 1
                # V index: (j, k) -> j*p + k
                V[j * p + k][r] = 1
                # W index: (k, i) -> k*n + i
                W[k * n + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
