"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    The output factor W uses the official trace convention W[k*n + i, r].
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    rank = 7

    # U: shape (n*m, rank) = (4, 7)
    # Rows correspond to A[i, j] with index i*m + j
    U = [
        [1,  0,  1,  0,  1, -1,  0],  # (0,0)
        [0,  0,  0,  0,  1,  0,  1],  # (0,1)
        [0,  1,  0,  0,  0,  1,  0],  # (1,0)
        [1,  1,  0,  1,  0,  0, -1],  # (1,1)
    ]

    # V: shape (m*p, rank) = (4, 7)
    # Rows correspond to B[j, k] with index j*p + k
    V = [
        [1,  1,  0, -1,  0,  1,  0],  # (0,0)
        [0,  0,  1,  0,  0,  1,  0],  # (0,1)
        [0,  0,  0,  1,  0,  0,  1],  # (1,0)
        [1,  0, -1, 0,  1,  0,  1],  # (1,1)
    ]

    # W: shape (p*n, rank) = (4, 7)
    # Rows correspond to C[i, k] with index k*n + i (trace convention)
    # i is row index of A, k is column index of B
    # W[k*n + i] = sum_r U[i*m+j, r] * V[j*p+k, r]
    W = [
        [1,  0,  0,  1, -1,  0,  1],  # k=0, i=0 -> C_00
        [0,  1,  0,  1,  0,  0,  0],  # k=0, i=1 -> C_10
        [0,  0,  1,  0,  1,  0,  0],  # k=1, i=0 -> C_01
        [1, -1,  1,  0,  0,  1,  0],  # k=1, i=1 -> C_11
    ]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
