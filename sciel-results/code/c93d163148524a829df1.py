"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a rank 30 decomposition.
    # Base: 3x3x3 schoolbook decomposition (rank 27)
    # Extension: 3 terms for the 4th column of B (k=3)
    # Total rank: 27 + 3 = 30
    
    rank = 30
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Part 1: 3x3x3 Schoolbook (i in 0..2, j in 0..2, k in 0..2)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # Only first 3 columns of k
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    # Part 2: Extension for k=3
    # We need to compute C[i, 3] = sum_j A[i, j] B[j, 3]
    # This is decomposed into 3 terms, one for each j.
    for j in range(m):
        for i in range(n):
            # U[i, j] = 1
            U[i * m + j][col] = 1.0
            # V[j, 3] = 1
            V[j * p + 3][col] = 1.0
            # W[3, i] = 1
            W[3 * n + i][col] = 1.0
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
