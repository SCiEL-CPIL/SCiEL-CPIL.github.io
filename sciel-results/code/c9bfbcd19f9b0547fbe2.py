"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 2 * p  # 24
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 24 x 24
    
    col = 0
    for k in range(p):
        # Term 1: (A[0,0]+A[1,0]) * B[0,k] -> contributes to C[0,k] and C[1,k]
        U[0][col] = 1   # A[0,0]
        U[2][col] = 1   # A[1,0]
        V[0 * p + k][col] = 1  # B[0,k]
        W[k * n + 0][col] = 1  # C[0,k]
        W[k * n + 1][col] = 1  # C[1,k]
        col += 1
        
        # Term 2: (A[0,1]+A[1,1]) * B[1,k] -> contributes to C[0,k] and C[1,k]
        U[1][col] = 1   # A[0,1]
        U[3][col] = 1   # A[1,1]
        V[1 * p + k][col] = 1  # B[1,k]
        W[k * n + 0][col] = 1  # C[0,k]
        W[k * n + 1][col] = 1  # C[1,k]
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
