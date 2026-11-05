"""Matrix multiplication 3x4x6 - rank 24 decomposition."""
DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 4 * 6 = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            # u[i] = 1 for all i
            for i in range(n):
                U[i * m + j][col] = 1
            # v[j'] = 1 if j' == j
            V[j * p + k][col] = 1
            # w[k'] = 1 if k' == k  (W index is k*n + i, but w only depends on k)
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
