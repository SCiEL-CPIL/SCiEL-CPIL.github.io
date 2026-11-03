"""Tensor decomposition for 2x4 * 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m  # 4
    U = [[0] * rank for _ in range(n * m)]  # 8 x 4
    V = [[0] * rank for _ in range(m * p)]  # 24 x 4
    W = [[0] * rank for _ in range(p * n)]  # 12 x 4
    
    for t in range(rank):
        for i in range(n):
            U[i * m + t][t] = 1  # U[i*m+j, t] = 1 if j==t
        for j in range(m):
            for k in range(p):
                if j == t and k == t:
                    V[j * p + k][t] = 1  # V[j*p+k, t] = 1 if j==t and k==t
        for k in range(p):
            for i in range(n):
                if k == t:
                    W[k * n + i][t] = 1  # W[k*n+i, t] = 1 if k==t
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
