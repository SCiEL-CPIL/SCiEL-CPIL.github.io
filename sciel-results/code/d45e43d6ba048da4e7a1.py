"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 12 decomposition for 2x2x6 matrix multiplication
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        # Term 2k: alpha = 0
        col0 = 2 * k
        # u = (1, 1, 0, 0)
        U[0][col0] = 1
        U[1][col0] = 1
        # v = e_k + e_{6+k}
        V[k][col0] = 1
        V[6 + k][col0] = 1
        # W coefficients for S_{2k} and S_{2k+1}
        # S_{2k} = 1 * M0 + 0 * M1
        W[2 * k][col0] = 1
        # S_{2k+1} = -1 * M0 + 1 * M1
        W[2 * k + 1][col0] = -1
        
        # Term 2k+1: alpha = 1
        col1 = 2 * k + 1
        # u = (1, 1, 1, 1)
        U[0][col1] = 1
        U[1][col1] = 1
        U[2][col1] = 1
        U[3][col1] = 1
        # v = e_k + e_{6+k}
        V[k][col1] = 1
        V[6 + k][col1] = 1
        # W coefficients
        W[2 * k][col1] = 0
        W[2 * k + 1][col1] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
