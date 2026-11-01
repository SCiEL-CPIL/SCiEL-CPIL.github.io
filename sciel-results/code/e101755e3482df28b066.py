"""Matrix multiplication 2x2 x 2x16 via Strassen decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    Uses Strassen's rank-7 decomposition for the 2x2 core, applied
    independently to each of the 8 column-pairs of the 2x16 matrix.
    Total rank = 7 * 8 = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen 2x2x2 decomposition (inline for speed)
    U_core = np.zeros((4, 7), dtype=np.int64)
    V_core = np.zeros((4, 7), dtype=np.int64)
    W_core = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h)
    U_core[0, 0] = 1; U_core[3, 0] = 1
    V_core[0, 0] = 1; V_core[3, 0] = 1
    W_core[0, 0] = 1; W_core[3, 0] = 1
    
    # M2 = (c+d)e
    U_core[2, 1] = 1; U_core[3, 1] = 1
    V_core[0, 1] = 1
    W_core[2, 1] = 1; W_core[3, 1] = -1
    
    # M3 = a(f-h)
    U_core[0, 2] = 1
    V_core[1, 2] = 1; V_core[3, 2] = -1
    W_core[1, 2] = 1; W_core[3, 2] = 1
    
    # M4 = d(g-e)
    U_core[3, 3] = 1
    V_core[2, 3] = 1; V_core[0, 3] = -1
    W_core[0, 3] = 1; W_core[2, 3] = 1
    
    # M5 = (a+b)h
    U_core[0, 4] = 1; U_core[1, 4] = 1
    V_core[3, 4] = 1
    W_core[0, 4] = -1; W_core[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U_core[2, 5] = 1; U_core[0, 5] = -1
    V_core[0, 5] = 1; V_core[1, 5] = 1
    W_core[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U_core[1, 6] = 1; U_core[3, 6] = -1
    V_core[2, 6] = 1; V_core[3, 6] = 1
    W_core[0, 6] = 1
    
    num_pairs = p // 2  # 8 pairs
    rank = 7 * num_pairs  # 56
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for t in range(num_pairs):
        col_offset = t * 7
        
        for r in range(7):
            U[:, col_offset + r] = U_core[:, r]
            V[2*t, col_offset + r] = V_core[0, r]
            V[2*t + 1, col_offset + r] = V_core[1, r]
            V[16 + 2*t, col_offset + r] = V_core[2, r]
            V[16 + 2*t + 1, col_offset + r] = V_core[3, r]
            W[4*t, col_offset + r] = W_core[0, r]
            W[4*t + 2, col_offset + r] = W_core[1, r]
            W[4*t + 1, col_offset + r] = W_core[2, r]
            W[4*t + 3, col_offset + r] = W_core[3, r]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
