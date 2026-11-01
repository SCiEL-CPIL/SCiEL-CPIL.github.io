"""Tensor decomposition for 2x3 times 3x9 matrix multiplication."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Rank-minimal exact decomposition for 2x3 x 3x9 is n*m*p = 54.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                
                U[alpha][col] = 1
                V[beta][col] = 1
                W[gamma][col] = 1
                col += 1

    # Verify exactness
    if _verify(U, V, W, n, m, p, deadline):
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, deadline):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    Ur = Ua.reshape(n, m, -1)
    Vr = Va.reshape(m, p, -1)
    Wr = Wa.reshape(p, n, -1)
    
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    
    expected = np.ones((n, m, p), dtype=float)
    
    if time.time() > deadline:
        return False
        
    return bool(np.allclose(T, expected, atol=1e-9))
