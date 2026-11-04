"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 5, 5]
_DEADLINE = 120.0  # wall-clock seconds

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 @ 5x5 matrix multiplication.
    
    Uses the schoolbook decomposition with rank n*m*p = 75.
    For each triple (i, j, k) with i in [0,n), j in [0,m), k in [0,p):
      U[i*m + j, col] = 1
      V[j*p + k, col] = 1
      W[k*n + i, col] = 1
    This exactly decomposes the matrix multiplication tensor.
    """
    _start = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p  # 75
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                # Deadline guard: if we're running out of time, return what we have
                if time.time() - _start > _DEADLINE - 5.0:
                    # Truncate to complete columns only
                    valid_cols = column
                    U = U[:, :valid_cols]
                    V = V[:, :valid_cols]
                    W = W[:, :valid_cols]
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
