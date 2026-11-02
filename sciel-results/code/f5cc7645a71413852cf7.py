import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    The tensor rank of 2x2 matrix multiplication is known to be 7 (Strassen).
    Rank 6 is impossible over any field.
    
    Interface:
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank) with W[k*n+i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U[i*m + a, r] : Coefficient of A[i,a] in M_r
    # V[a*p + k, r] : Coefficient of B[a,k] in M_r
    # W[k*n + i, r] : Coefficient of M_r in C[i,k]
    
    # M1: (A00 + A11)(B00 + B11)
    U[0*2+0, 0] = 1  # A00
    U[1*2+1, 0] = 1  # A11
    V[0*2+0, 0] = 1  # B00
    V[1*2+1, 0] = 1  # B11
    
    # M2: (A10 + A11)B00
    U[1*2+0, 1] = 1  # A10
    U[1*2+1, 1] = 1  # A11
    V[0*2+0, 1] = 1  # B00
    
    # M3: A00(B01 - B11)
    U[0*2+0, 2] = 1  # A00
    V[0*2+1, 2] = 1  # B01
    V[1*2+1, 2] = -1 # -B11
    
    # M4: A11(B10 - B00)
    U[1*2+1, 3] = 1  # A11
    V[1*2+0, 3] = 1  # B10
    V[0*2+0, 3] = -1 # -B00
    
    # M5: (A00 + A01)B11
    U[0*2+0, 4] = 1  # A00
    U[0*2+1, 4] = 1  # A01
    V[1*2+1, 4] = 1  # B11
    
    # M6: (A10 - A00)(B00 + B01)
    U[1*2+0, 5] = 1  # A10
    U[0*2+0, 5] = -1 # -A00
    V[0*2+0, 5] = 1  # B00
    V[0*2+1, 5] = 1  # B01
    
    # M7: (A01 - A11)(B10 + B11)
    U[0*2+1, 6] = 1  # A01
    U[1*2+1, 6] = -1 # -A11
    V[1*2+0, 6] = 1  # B10
    V[1*2+1, 6] = 1  # B11
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    W[0*2+0, 0] = 1   # M1
    W[0*2+0, 3] = 1   # M4
    W[0*2+0, 4] = -1  # -M5
    W[0*2+0, 6] = 1   # M7
    
    # C01 = M3 + M5
    W[1*2+0, 2] = 1   # M3
    W[1*2+0, 4] = 1   # M5
    
    # C10 = M2 + M4
    W[0*2+1, 1] = 1   # M2
    W[0*2+1, 3] = 1   # M4
    
    # C11 = M1 - M2 + M3 + M6
    W[1*2+1, 0] = 1   # M1
    W[1*2+1, 1] = -1  # -M2
    W[1*2+1, 2] = 1   # M3
    W[1*2+1, 5] = 1   # M6
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
