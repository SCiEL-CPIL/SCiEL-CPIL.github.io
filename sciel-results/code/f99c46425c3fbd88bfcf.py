"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 2 * p  # 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        # Term 2k: diagonal part (i == j)
        r = 2 * k
        for i in range(n):
            U[i * m + i][r] = 1  # diagonal: i==j
            W[k * n + i][r] = 1
        for j in range(m):
            V[j * p + k][r] = 1
        
        # Term 2k+1: off-diagonal part (i == 1-j)
        r = 2 * k + 1
        for i in range(n):
            j = 1 - i  # off-diagonal
            U[i * m + j][r] = 1
            W[k * n + i][r] = 1
        for j in range(m):
            V[j * p + k][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
