"""Matrix multiplication 3,3,7 - rank-minimal exact tensor decomposition."""

import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor T[i,j,k] = delta."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reconstruct tensor: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m + j, j*p + k, k*n + i] = 1.0
    
    return np.allclose(recon, target, atol=tol)

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return rank, U, V, W

def construct():
    """
    Returns a valid rank decomposition for 3x3 by 3x7 matrix multiplication.
    Uses schoolbook (rank 63).
    """
    n, m, p = 3, 3, 7
    rank, U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
