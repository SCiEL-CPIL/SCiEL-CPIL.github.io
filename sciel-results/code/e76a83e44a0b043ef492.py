"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W give an exact decomposition of the matrix multiplication tensor."""
    rank = U.shape[1]
    
    # Check all target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check some non-target entries
    # a=0 (i=0,j=0), b=0 (j=0,k=0): target c=0, so c=1 should be 0
    val = np.sum(U[0, :] * V[0, :] * W[1, :])
    if abs(val) > tol:
        return False
    
    # a=1 (i=0,j=1), b=0 (j=0,k=0): j mismatch, so no target
    val = np.sum(U[1, :] * V[0, :] * W[0, :])
    if abs(val) > tol:
        return False
    
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, 20s safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify(U, V, W, n, m, p):
        raise Exception("Schoolbook verification failed!")
    
    # Return the schoolbook decomposition
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
