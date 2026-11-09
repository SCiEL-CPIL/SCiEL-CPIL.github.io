"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # The 2x3 * 3x8 matrix multiplication tensor has entry 1 at every
    # (i, j, k) triple with i in [0,n), j in [0,m), k in [0,p).
    # This is the all-ones tensor on the index set [0,n)x[0,m)x[0,p),
    # which is a rank-1 tensor: the outer product of three all-ones vectors.
    #
    # Trace convention: W[k*n+i, r]
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1] * rank for _ in range(n * m)]   # shape (n*m, 1) = (6, 1)
    V = [[1] * rank for _ in range(m * p)]   # shape (m*p, 1) = (24, 1)
    W = [[1] * rank for _ in range(p * n)]   # shape (p*n, 1) = (16, 1)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
