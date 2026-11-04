"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # since C[i,k] = sum_j A[i,j]*B[j,k] includes every (i,j,k) term.
    # Rank-1 decomposition: U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
