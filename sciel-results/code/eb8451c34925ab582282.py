"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition (optimal rank for 2x2).
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    rank = 7

    # Strassen's 7 products coefficients
    # U[i*m+j, c]: coefficient of A[i,j] in product c
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1]
    ]

    # V[j*p+k, c]: coefficient of B[j,k] in product c
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],   # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]   # B[1,1]
    ]

    # W[k*n+i, c]: coefficient for C[i,k] from product c
    # Row index = k*n + i
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0] (k=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0] (k=0, i=1)
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1] (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]   # C[1,1] (k=1, i=1)
    ]

    U = np.array(U_data, dtype=float)
    V = np.array(V_data, dtype=float)
    W = np.array(W_data, dtype=float)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
