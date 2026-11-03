"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor for n x m by m x p is the all-ones tensor.
    # T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # This tensor has rank 1, as it is the outer product of three all-ones vectors.
    rank = 1
    
    # U shape (n*m, rank)
    U = [[1] * rank for _ in range(n * m)]
    # V shape (m*p, rank)
    V = [[1] * rank for _ in range(m * p)]
    # W shape (p*n, rank)
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
