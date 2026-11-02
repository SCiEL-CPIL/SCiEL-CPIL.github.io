"""Tensor decomposition for 2x2 matrix multiplication (Strassen, rank 7)."""
import numpy as np

def construct():
    """
    Returns Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    The rank is 7, which is the proven theoretical minimum.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    Shapes: U (4, 7), V (4, 7), W (4, 7).
    Convention:
      U[i*m + j, c] for A[i, j]
      V[j'*p + k, c] for B[j', k]
      W[k'*n + i, c] for C[i, k]
      with n = m = p = 2.
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # m0 = (a00+a11)(b00+b11)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0

    # m1 = (a10+a11)b00
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[1, 1] = 1.0; W[3, 1] = -1.0

    # m2 = a00(b01-b11)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[2, 2] = 1.0; W[3, 2] = 1.0

    # m3 = a11(b10-b00)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[1, 3] = 1.0

    # m4 = (a00+a01)b11
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[2, 4] = 1.0

    # m5 = (a10-a00)(b00+b01)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0

    # m6 = (a01-a11)(b10+b11)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
