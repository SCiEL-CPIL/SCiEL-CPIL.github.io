"""Matrix multiplication 2,3,10 - search for rank-minimal decomposition."""
import numpy as np
import time
import random
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-6):
    """Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1.0
                if abs(val - expected) > tol:
                    return False
    return True

def _verify_fast(U, V, W, tol=1e-6):
    """Faster verification using numpy."""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=tol)

def _construct_rank_30():
    """
    Construct a rank-30 decomposition for 2x3x10 matrix multiplication.
    This exploits the lower bound max(nm, mp, pn) = 30.
    """
    n, m, p = DIMENSIONS
    rank = m * p  # 30
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Index c corresponds to (j, k) where j in [0, m), k in [0, p)
    # c = j * p + k
    for j in range(m):
        for k in range(p):
            c = j * p + k
            
            # V is identity-like: V[j*p+k, c] = 1
            V[j * p + k][c] = 1.0
            
            # U[i*m+j, c] = 1 for all i
            for i in range(n):
                U[i * m + j][c] = 1.0
                
            # W[k*n+i, c] = 1 for all i
            for i in range(n):
                W[k * n + i][c] = 1.0
                
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    # The rank-30 construction in _construct_rank_30 is invalid for the
    # standard matrix multiplication tensor support because it does not
    # enforce the index matching i == i' in the output factor W.
    # Therefore, we fall back to the schoolbook decomposition which is
    # guaranteed to be correct.
    
    # Attempt to verify schoolbook just in case (it should always pass)
    result = _schoolbook()
    
    if _verify_fast(result["U"], result["V"], result["W"], tol=1e-9):
        return result
        
    # If even schoolbook fails verification (should not happen), return it anyway
    # as it is the standard correct decomposition.
    return result
# EVOLVE-BLOCK-END
