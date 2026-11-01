"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Theoretical minimal rank for 2x2x1 is 4, so for 2x2x14 it is 4*p
    rank = 4 * p
    
    # Initialize U, V, W as zero matrices
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # We generate 4 terms per column k
    # Term 1: A[0,0] * B[0,k] -> C[0,k]
    # Term 2: A[0,1] * B[1,k] -> C[0,k]
    # Term 3: A[1,0] * B[0,k] -> C[1,k]
    # Term 4: A[1,1] * B[1,k] -> C[1,k]
    
    col_idx = 0
    for k in range(p):
        # Term 1
        U[0 * m + 0, col_idx] = 1.0
        V[0 * p + k, col_idx] = 1.0
        W[k * n + 0, col_idx] = 1.0
        col_idx += 1
        
        # Term 2
        U[0 * m + 1, col_idx] = 1.0
        V[1 * p + k, col_idx] = 1.0
        W[k * n + 0, col_idx] = 1.0
        col_idx += 1
        
        # Term 3
        U[1 * m + 0, col_idx] = 1.0
        V[0 * p + k, col_idx] = 1.0
        W[k * n + 1, col_idx] = 1.0
        col_idx += 1
        
        # Term 4
        U[1 * m + 1, col_idx] = 1.0
        V[1 * p + k, col_idx] = 1.0
        W[k * n + 1, col_idx] = 1.0
        col_idx += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
