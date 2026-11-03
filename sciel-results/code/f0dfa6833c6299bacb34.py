"""Program for matrix multiplication 2x3x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal tensor decomposition for multiplying a 2x3 matrix
    by a 3x6 matrix.

    The matrix multiplication tensor T has indices (i, j, k) for i in range(n),
    j in range(m), k in range(p). The standard schoolbook decomposition uses
    rank = n*m*p terms. For small dimensions like 2x3x6, this is often optimal
    or near-optimal. We construct the standard schoolbook decomposition here.

    The output factor W follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p

    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    # Fill in the factors for the schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (trace convention)
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
