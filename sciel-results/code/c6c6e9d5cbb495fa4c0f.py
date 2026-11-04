"""Rewritten program using NumPy reshape/broadcast approach for distinct structure."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    R = n * m * p
    # Build U: shape (n*m, R), where column indexed by (i,j,k) -> j*p + k? No.
    # We need column index for each (i,j,k). Let's use col = (i*m + j)*p + k
    # Then U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    
    # Generate all column indices for each (i,j,k)
    ii = np.repeat(np.arange(n), m * p).reshape(n, m, p)
    jj = np.tile(np.repeat(np.arange(m), p), (n, 1)).reshape(n, m, p)
    kk = np.tile(np.arange(p), (n * m,)).reshape(n, m, p)
    
    col_idx = (ii * m + jj) * p + kk  # shape (n, m, p)
    col_idx_flat = col_idx.reshape(-1)
    
    rowU = (ii * m + jj).reshape(-1)  # i*m+j
    rowV = (jj * p + kk).reshape(-1)  # j*p+k
    rowW = (kk * n + ii).reshape(-1)  # k*n+i
    
    U = np.zeros((n * m, R), dtype=np.int64)
    V = np.zeros((m * p, R), dtype=np.int64)
    W = np.zeros((p * n, R), dtype=np.int64)
    
    U[rowU, col_idx_flat] = 1
    V[rowV, col_idx_flat] = 1
    W[rowW, col_idx_flat] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
