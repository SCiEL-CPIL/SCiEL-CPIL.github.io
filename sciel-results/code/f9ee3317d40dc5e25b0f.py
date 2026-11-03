"""Matrix multiplication 2x3 x 3x10: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-55 exact tensor decomposition for 2x3 x 3x10 matrix multiplication.
    
    Structure:
    - Part 1 (rank 20): schoolbook for A[:,2] (2x1) * B[2,:] (1x10)
    - Part 2 (rank 35): Strassen decomposition for A[:,:2] (2x2) * B[:2,:] (2x10)
      using 5 independent 2x2x2 blocks with Strassen's 7-term algorithm.
    
    Total rank = 55.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 55
    
    U = np.zeros((n * m, rank))  # (6, 55)
    V = np.zeros((m * p, rank))  # (30, 55)
    W = np.zeros((p * n, rank))  # (20, 55)
    
    col = 0
    
    # Part 1: A[:,2] * B[2,:] — schoolbook, rank 20
    # A indices: A[0,2]=2, A[1,2]=5
    # B indices: B[2,k] = 20+k for k=0..9
    # C indices: C[i,k] at W[k*n+i] = W[2k+i]
    for i in range(n):
        a_idx = i * m + 2
        for k in range(p):
            b_idx = 2 * p + k
            c_idx = k * n + i
            U[a_idx, col] = 1.0
            V[b_idx, col] = 1.0
            W[c_idx, col] = 1.0
            col += 1
    
    # Part 2: A[:,:2] * B[:2,:] — Strassen, rank 35
    # A indices in U: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B indices in V: B[0,k]=k, B[1,k]=10+k
    # C indices in W: C[i,k] at W[2k+i]
    
    X00, X01, X10, X11 = 0, 1, 3, 4
    
    for j in range(5):
        # B block indices for columns 2j, 2j+1
        Y00 = 2 * j
        Y01 = 2 * j + 1
        Y10 = 10 + 2 * j
        Y11 = 10 + 2 * j + 1
        
        # C block indices
        Z00 = 4 * j      # C[0, 2j]
        Z10 = 4 * j + 1  # C[1, 2j]
        Z01 = 4 * j + 2  # C[0, 2j+1]
        Z11 = 4 * j + 3  # C[1, 2j+1]
        
        # Strassen's 7 products for 2x2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M3
        # C01 = M3 + M5
        # C11 = M1 - M2 + M4 + M6
        
        # M1
        U[X00, col] = 1.0
        U[X11, col] = 1.0
        V[Y00, col] = 1.0
        V[Y11, col] = 1.0
        W[Z00, col] = 1.0
        W[Z11, col] = 1.0
        col += 1
        
        # M2
        U[X10, col] = 1.0
        U[X11, col] = 1.0
        V[Y00, col] = 1.0
        W[Z10, col] = 1.0
        W[Z11, col] = -1.0
        col += 1
        
        # M3
        U[X00, col] = 1.0
        V[Y01, col] = 1.0
        V[Y11, col] = -1.0
        W[Z01, col] = 1.0
        W[Z11, col] = 1.0
        col += 1
        
        # M4
        U[X11, col] = 1.0
        V[Y10, col] = 1.0
        V[Y00, col] = -1.0
        W[Z00, col] = 1.0
        W[Z10, col] = 1.0
        col += 1
        
        # M5
        U[X00, col] = 1.0
        U[X01, col] = 1.0
        V[Y11, col] = 1.0
        W[Z00, col] = -1.0
        W[Z01, col] = 1.0
        col += 1
        
        # M6
        U[X10, col] = 1.0
        U[X00, col] = -1.0
        V[Y00, col] = 1.0
        V[Y01, col] = 1.0
        W[Z11, col] = 1.0
        col += 1
        
        # M7
        U[X01, col] = 1.0
        U[X11, col] = -1.0
        V[Y10, col] = 1.0
        V[Y11, col] = 1.0
        W[Z00, col] = 1.0
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
