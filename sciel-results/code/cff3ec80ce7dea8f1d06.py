"""Program for matrix multiplication tensor decomposition (3x4x5)."""

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Uses the standard schoolbook algorithm which is rank n*m*p.
    """
    import numpy as np

    n, m, p = DIMENSIONS
    rank = n * m * p

    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Map (i,j,k) to column index c
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k

                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1

    # Convert to lists as per interface expectation (if strict) or return arrays
    # The task says "factor matrices", lists are safer if not specified as arrays
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
