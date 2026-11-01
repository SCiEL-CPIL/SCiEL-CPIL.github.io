import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-32 decomposition of the 2x2 by 2x16 matrix multiplication tensor.
    
    Structure:
    - n=2, m=2, p=16
    - U: shape (4, 32) — factors for A (2x2)
    - V: shape (32, 32) — factors for B (2x16)
    - W: shape (32, 32) — factors for C (2x16), indexed as W[k*n+i, r] = W[16*k+i, r]
    
    For each column k of B (0..15), we need:
      C[0, k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
      C[1, k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    We use 2 terms per column (rank 32 total):
      Term 2k:   U picks A[0,0] and A[1,0]; V picks B[0,k]; W picks C[0,k] and C[1,k]
      Term 2k+1: U picks A[0,1] and A[1,1]; V picks B[1,k]; W picks C[0,k] and C[1,k]
    
    This gives:
      C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]  ✓
      C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]  ✓
    """
    n, m, p = DIMENSIONS
    rank = 2 * p  # 32
    
    # U: (n*m, rank) = (4, 32)
    # Index: i*m + j where i is row of A, j is col of A
    # A[0,0] -> 0, A[0,1] -> 1, A[1,0] -> 2, A[1,1] -> 3
    U = np.zeros((n * m, rank), dtype=int)
    
    # V: (m*p, rank) = (32, 32)
    # Index: j*p + k where j is row of B, k is col of B
    # B[0,k] -> k, B[1,k] -> p + k
    V = np.zeros((m * p, rank), dtype=int)
    
    # W: (p*n, rank) = (32, 32)
    # Index: k*n + i where k is col of C, i is row of C
    # C[i,k] -> k*n + i
    W = np.zeros((p * n, rank), dtype=int)
    
    for k in range(p):
        # Term 1 (even rank index): use A[0,0] and A[1,0] with B[0,k]
        r1 = 2 * k
        U[0, r1] = 1   # A[0,0]
        U[2, r1] = 1   # A[1,0]
        V[k, r1] = 1   # B[0,k]
        W[k * n + 0, r1] = 1  # C[0,k]
        W[k * n + 1, r1] = 1  # C[1,k]
        
        # Term 2 (odd rank index): use A[0,1] and A[1,1] with B[1,k]
        r2 = 2 * k + 1
        U[1, r2] = 1   # A[0,1]
        U[3, r2] = 1   # A[1,1]
        V[p + k, r2] = 1  # B[1,k]
        W[k * n + 0, r2] = 1  # C[0,k]
        W[k * n + 1, r2] = 1  # C[1,k]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
