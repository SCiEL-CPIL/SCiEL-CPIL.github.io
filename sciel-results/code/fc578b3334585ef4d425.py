"""Find rank-minimal tensor decomposition for 2x5 * 5x6 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=2, m=5, p=6):
    """Verify that U, V, W form a valid tensor decomposition for matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check the "1" entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check all "0" entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i == i2 and j == j2 and k == k2:
                                continue
                            b = j2 * p + k2
                            c = k2 * n + i2
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            if abs(val) > 1e-9:
                                return False
    return True


def _schoolbook_rank(n=2, m=5, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2 by 5 matrix by a 5 by 6 matrix.
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Return schoolbook decomposition
    best = _schoolbook_rank(n, m, p)
    
    # Verify it's correct
    assert _verify_decomposition(best["U"], best["V"], best["W"], n, m, p), "Schoolbook decomposition failed verification"
    
    return best
# EVOLVE-BLOCK-END
