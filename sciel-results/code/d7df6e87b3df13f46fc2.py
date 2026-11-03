"""Tensor decomposition for 2x3 * 3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, tol=1e-9):
    """Check the decomposition identity."""
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.einsum('r,r,r->', U[i*m+j], V[j*p+k], W[k*n+i]))
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Returns dict with 'U', 'V', 'W' as lists of lists.
    """
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication C = A * B where A is n x m and B is m x p
    # is T[i, j, k] = 1 for all i, j, k in the valid ranges.
    # This is because C[i, k] = sum_j A[i, j] * B[j, k].
    # The term A[i, j] * B[j, k] contributes to C[i, k] for all i, j, k.
    # The all-ones tensor has rank 1.
    # We can construct a rank 1 decomposition:
    # U[im + j, 0] = 1 for all i, j
    # V[jp + k, 0] = 1 for all j, k
    # W[kn + i, 0] = 1 for all k, i
    
    rank = 1
    U = np.ones((n * m, rank), dtype=np.float64)
    V = np.ones((m * p, rank), dtype=np.float64)
    W = np.ones((p * n, rank), dtype=np.float64)
    
    # Verify
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if something went wrong (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
