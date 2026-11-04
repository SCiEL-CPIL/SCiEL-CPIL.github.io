"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    Uses a simple schoolbook decomposition to ensure validity.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Schoolbook decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * U_rows
                v = [0] * V_rows
                w = [0] * W_rows
                
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
                
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
