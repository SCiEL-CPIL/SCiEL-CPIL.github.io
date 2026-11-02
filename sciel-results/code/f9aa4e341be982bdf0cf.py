"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T[i,j,k] = 1 for all valid i,j,k
    # (since A[i,j]*B[j,k] always contributes to C[i,k] with coefficient 1)
    # This all-ones tensor has tensor rank 1:
    # T[i,j,k] = U[i,0] * V[j,0] * W[k,0] where U,V,W are all-ones vectors
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
