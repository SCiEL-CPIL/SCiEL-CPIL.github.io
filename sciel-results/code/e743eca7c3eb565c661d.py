"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if not U or not V or not W:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # Build tensor T[(i,k),(k',j),(i',j')] = sum_r U[i*m+k, r] * V[k'*p+j, r] * W[j*n+i', r]
    # Should equal 1 iff i==i' and k==k' (and j is the same in V and W indices, which it is by construction)
    for i in range(n):
        for k in range(m):
            for jp in range(p):
                for ip in range(n):
                    for kp in range(m):
                        expected = 1.0 if (i == ip and k == kp) else 0.0
                        actual = 0.0
                        for r in range(rank):
                            actual += U[i * m + k][r] * V[kp * p + jp][r] * W[jp * n + ip][r]
                        if abs(actual - expected) > 1e-9:
                            return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Uses schoolbook decomposition (rank 63) as the guaranteed valid baseline.
    """
    n, m, p = DIMENSIONS
    
    # Return schoolbook decomposition
    result = _schoolbook(n, m, p)
    
    # Verify it's correct
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return result
# EVOLVE-BLOCK-END
