"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Strassen-based decomposition for 2x2 * 2x12 matrix multiplication.
    
    Splits the 12 columns into 6 pairs of 2 columns, applying Strassen's
    rank-7 algorithm to each 2x2x2 block. Total rank = 6 * 7 = 42.
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j (i in 0..n-1, j in 0..m-1) for A[i][j]
      V[j*p+k, r]: row = j*p+k (j in 0..m-1, k in 0..p-1) for B[j][k]
      W[k*n+i, r]: row = k*n+i (k in 0..p-1, i in 0..n-1) for C[i][k] (trace convention)
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strassen's 7 products for 2x2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # Output:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # U factors: (ai, aj, coeff) means U[ai*m + aj, col] = coeff
    strassen_U = [
        [(0,0, 1), (1,1, 1)],   # M0
        [(1,0, 1), (1,1, 1)],   # M1
        [(0,0, 1)],              # M2
        [(1,1, 1)],              # M3
        [(0,0, 1), (0,1, 1)],   # M4
        [(1,0, 1), (0,0,-1)],   # M5
        [(0,1, 1), (1,1,-1)],   # M6
    ]
    
    # V factors: (bj, bk, coeff) means V[bj*p + bk, col] = coeff (bk is relative to block)
    strassen_V = [
        [(0,0, 1), (1,1, 1)],   # M0
        [(0,0, 1)],              # M1
        [(0,1, 1), (1,1,-1)],   # M2
        [(1,0, 1), (0,0,-1)],   # M3
        [(1,1, 1)],              # M4
        [(0,0, 1), (0,1, 1)],   # M5
        [(1,0, 1), (1,1, 1)],   # M6
    ]
    
    # W factors: (ci, ck, coeff) means W[ck*n + ci, col] = coeff (ck is relative to block)
    strassen_W = [
        [(0,0, 1), (1,1, 1)],   # M0
        [(1,0, 1), (1,1,-1)],   # M1
        [(0,1, 1), (1,1, 1)],   # M2
        [(0,0, 1), (1,0, 1)],   # M3
        [(0,0,-1), (0,1, 1)],   # M4
        [(1,1, 1)],              # M5
        [(0,0, 1)],              # M6
    ]
    
    num_blocks = p // 2  # 6 blocks of 2 columns each
    rank = num_blocks * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42
    
    col = 0
    for t in range(num_blocks):
        base_col = 2 * t  # starting column index in B and C
        for s in range(7):
            # Fill U
            for (ai, aj, coeff) in strassen_U[s]:
                u_row = ai * m + aj
                U[u_row][col] = coeff
            
            # Fill V (bk is relative offset within the 2-column block)
            for (bj, bk, coeff) in strassen_V[s]:
                b_col = base_col + bk
                v_row = bj * p + b_col
                V[v_row][col] = coeff
            
            # Fill W (ck is relative offset within the 2-column block)
            for (ci, ck, coeff) in strassen_W[s]:
                c_col = base_col + ck
                w_row = c_col * n + ci
                W[w_row][col] = coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
