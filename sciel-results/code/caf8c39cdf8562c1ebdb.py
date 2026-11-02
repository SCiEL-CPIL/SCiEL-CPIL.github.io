"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 2 * p  # 22, optimal for (2,2,11)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        r1 = 2 * k
        U[0][r1] = 1
        U[1][r1] = 1
        V[k][r1] = 1
        V[p + k][r1] = 1
        W[2 * k][r1] = 1
        
        r2 = 2 * k + 1
        U[2][r2] = 1
        U[3][r2] = 1
        V[k][r2] = 1
        V[p + k][r2] = 1
        W[2 * k + 1][r2] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
