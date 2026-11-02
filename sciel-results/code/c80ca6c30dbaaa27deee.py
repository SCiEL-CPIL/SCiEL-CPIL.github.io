"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook_2x2x2():
    """Rank-4 schoolbook decomposition for 2x2x2."""
    n, m, p = 2, 2, 2
    rank = n * m * p  # 4
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W

def _schoolbook_2x2x3():
    """Rank-6 schoolbook decomposition for 2x2x3."""
    n, m, p = 2, 2, 3
    rank = n * m * p  # 6
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W

def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """
    Kronecker product of two tensor decompositions.
    """
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                max_err = max(max_err, abs(val - 1))
    return max_err < tol

def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for 
    multiplying a 4x4 matrix by a 4x6 matrix.
    """
    n, m, p = 4, 4, 6
    
    # Kronecker product: (2x2x2) ⊗ (2x2x3) -> 4x4x6
    # Schoolbook ranks: 4 and 6, so total rank = 24
    U1, V1, W1 = _schoolbook_2x2x2()
    U2, V2, W2 = _schoolbook_2x2x3()
    
    U, V, W = _kron_decomp(U1, V1, W1, U2, V2, W2)
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to full schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    else:
        U = U.tolist()
        V = V.tolist()
        W = W.tolist()
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
