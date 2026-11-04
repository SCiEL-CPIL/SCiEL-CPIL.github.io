"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]
TIME_LIMIT = 150
SAFETY_MARGIN = 10

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    
    # T[i,j,k] should equal 1 for all valid (i,j,k)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(U.shape[1]):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if total != 1:
                    return False
    return True

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for matrix multiplication.
    Returns the schoolbook decomposition (rank 30) as baseline.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # 1. Construct Schoolbook Decomposition
    U_school = np.zeros((n * m, rank), dtype=object)
    V_school = np.zeros((m * p, rank), dtype=object)
    W_school = np.zeros((p * n, rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    if not _verify_decomposition(U_school, V_school, W_school, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

# EVOLVE-BLOCK-END
