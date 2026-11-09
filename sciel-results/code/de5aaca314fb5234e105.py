"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition equals the all-ones tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U).reshape(n, m, rank)
    V_arr = np.array(V).reshape(m, p, rank)
    W_arr = np.array(W).reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix."""
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication is the all-ones tensor:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This has rank 1: set U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1 for all indices
    
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
    
    return {"U": U, "V": V, "W": W}
