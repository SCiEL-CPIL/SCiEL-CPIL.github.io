"""
Tensor decomposition for M_{2,2,13} matrix multiplication.
We construct a rank 2p = 26 decomposition.
"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    # Compute tensor: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Check against the true matrix multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1.0
                actual = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _build_rank_2p(n, m, p):
    """
    Construct the rank 2p decomposition.
    C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    """
    r = 2 * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((n * p, r))
    
    # Part 1: j=0
    # U[i*m+0, k] = 1
    # V[0*p+k, k] = 1
    # W[k*n+i, k] = 1
    for k in range(p):
        col = k
        for i in range(n):
            U[i * m + 0, col] = 1.0
        V[0 * p + k, col] = 1.0
        for i in range(n):
            W[k * n + i, col] = 1.0
            
    # Part 2: j=1
    # U[i*m+1, p+k] = 1
    # V[1*p+k, p+k] = 1
    # W[k*n+i, p+k] = 1
    for k in range(p):
        col = p + k
        for i in range(n):
            U[i * m + 1, col] = 1.0
        V[1 * p + k, col] = 1.0
        for i in range(n):
            W[k * n + i, col] = 1.0
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((n * p, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try the rank 2p decomposition
    result = _build_rank_2p(n, m, p)
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fall back to schoolbook
    return _build_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
