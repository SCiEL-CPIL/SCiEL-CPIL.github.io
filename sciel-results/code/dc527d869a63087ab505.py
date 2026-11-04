"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 27, which is optimal for this case
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            for i in range(n):
                U[i * m + j][c] = 1.0
            V[j * p + k][c] = 1.0
            for i in range(n):
                W[k * n + i][c] = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
