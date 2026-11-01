"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    We use Strassen's algorithm for each pair of columns. Since p=12 is even,
    we have 6 pairs of columns, each requiring rank 7 (Strassen).
    Total rank = 6 * 7 = 42.
    
    The decomposition satisfies:
    For all i in [0,n), j in [0,m), k in [0,p):
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    where the tensor encodes C[i,k] = sum_j A[i,j] * B[j,k].
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Number of column pairs
    num_pairs = p // 2  # 6 pairs
    
    rank = num_pairs * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen's 7 intermediate products for 2x2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Then:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    for t in range(num_pairs):
        base = t * 7
        for s in range(7):
            c = base + s
            if s == 0:
                # M1 = (A00 + A11) * (B00 + B11)
                U[0][c] = 1  # A00
                U[3][c] = 1  # A11
                V[0*p + 2*t][c] = 1  # B[0, 2t]
                V[1*p + 2*t+1][c] = 1  # B[1, 2t+1]
                W[2*t*n + 0][c] = 1  # C[0, 2t]
                W[(2*t+1)*n + 1][c] = 1  # C[1, 2t+1]
                
            elif s == 1:
                # M2 = (A10 + A11) * B00
                U[2][c] = 1  # A10
                U[3][c] = 1  # A11
                V[0*p + 2*t][c] = 1  # B[0, 2t]
                W[2*t*n + 1][c] = 1  # C[1, 2t]
                W[(2*t+1)*n + 1][c] = -1  # C[1, 2t+1]
                
            elif s == 2:
                # M3 = A00 * (B01 - B11)
                U[0][c] = 1  # A00
                V[0*p + 2*t+1][c] = 1  # B[0, 2t+1]
                V[1*p + 2*t+1][c] = -1  # B[1, 2t+1]
                W[(2*t+1)*n + 0][c] = 1  # C[0, 2t+1]
                W[(2*t+1)*n + 1][c] = 1  # C[1, 2t+1]
                
            elif s == 3:
                # M4 = A11 * (B10 - B00)
                U[3][c] = 1  # A11
                V[1*p + 2*t][c] = 1  # B[1, 2t]
                V[0*p + 2*t][c] = -1  # B[0, 2t]
                W[2*t*n + 0][c] = 1  # C[0, 2t]
                W[2*t*n + 1][c] = 1  # C[1, 2t]
                
            elif s == 4:
                # M5 = (A00 + A01) * B11
                U[0][c] = 1  # A00
                U[1][c] = 1  # A01
                V[1*p + 2*t+1][c] = 1  # B[1, 2t+1]
                W[2*t*n + 0][c] = -1  # C[0, 2t]
                W[(2*t+1)*n + 0][c] = 1  # C[0, 2t+1]
                
            elif s == 5:
                # M6 = (A10 - A00) * (B00 + B01)
                U[2][c] = 1  # A10
                U[0][c] = -1  # A00
                V[0*p + 2*t][c] = 1  # B[0, 2t]
                V[0*p + 2*t+1][c] = 1  # B[0, 2t+1]
                W[(2*t+1)*n + 1][c] = 1  # C[1, 2t+1]
                
            elif s == 6:
                # M7 = (A01 - A11) * (B10 + B11)
                U[1][c] = 1  # A01
                U[3][c] = -1  # A11
                V[1*p + 2*t][c] = 1  # B[1, 2t]
                V[1*p + 2*t+1][c] = 1  # B[1, 2t+1]
                W[2*t*n + 0][c] = 1  # C[0, 2t]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
