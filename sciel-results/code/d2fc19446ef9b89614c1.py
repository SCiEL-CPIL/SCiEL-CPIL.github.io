<<<<<<< SEARCH
"""Tensor decomposition for 2x3x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _verify_exact(U, V, W, n=2, m=3, p=6, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True
=======
"""Tensor decomposition for 2x3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _round_to_half_int(arr):
    """Round a numpy array to the nearest half-integer."""
    return np.round(arr * 2.0) / 2.0

def _is_half_integer(arr, tol=1e-9):
    """Check if all elements are close to integers or half-integers."""
    doubled = arr * 2.0
    return np.allclose(doubled, np.round(doubled), atol=tol)

def _verify_exact(U, V, W, n=2, m=3, p=6, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check half-integer constraint
    if not _is_half_integer(U, tol) or not _is_half_integer(V, tol) or not _is_half_integer(W, tol):
        return False
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True
>>>>>>> REPLACE

