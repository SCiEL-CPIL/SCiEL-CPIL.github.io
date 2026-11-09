DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = 2, 4, 7
    # Rank is n * m = 2 * 4 = 8?
    # For 1 x m by m x p, rank is m.
    # For n x m by m x p, rank is n * m.
    rank = n * m
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            # Term for row i, column j
            # U[i*m + j][col] = 1
            U[i*m + j][col] = 1
            # V[j*p + k][col] = 1 for all k
            for k in range(p):
                V[j*p + k][col] = 1
            # W[k*n + i][col] = 1 for all k
            for k in range(p):
                W[k*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
