"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank = n * p = 20
    rank = n * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Each term c corresponds to an (i0, k0) pair
    for i0 in range(n):
        for k0 in range(p):
            c = k0 * n + i0  # index for (i0, k0)
            # W[k0*n+i0, c] = 1
            W[k0 * n + i0][c] = 1
            # U[i0*m+j, c] = 1 for all j
            for j in range(m):
                U[i0 * m + j][c] = 1
            # V[j*p+k0, c] = 1 for all j
            for j in range(m):
                V[j * p + k0][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
