"""Tensor decomposition for 2x2 by 2x16 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix A by a 2x16 matrix B to produce a 2x16 matrix C.
    
    Uses Strassen's rank-7 decomposition for 2x2x2 multiplication,
    applied independently to each of the 8 column-pairs of B.
    Total rank = 7 * 8 = 56.
    
    Indexing conventions (per task contract):
    - U: shape (n*m, rank) = (4, 56), row index = i*m + j
    - V: shape (m*p, rank) = (32, 56), row index = j*p + k  
    - W: shape (p*n, rank) = (32, 56), row index = k*n + i
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen 2x2x2 core decomposition
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e)
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    
    # U_core: rows 0=a, 1=b, 2=c, 3=d
    # V_core: rows 0=e, 1=f, 2=g, 3=h
    # W_core: rows 0=C11, 1=C12, 2=C21, 3=C22
    
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
    
    num_pairs = p // 2  # 8 pairs of columns
    rank = 7 * num_pairs  # 56
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for t in range(num_pairs):
        col_off = t * 7
        # U: same pattern for all blocks
        U[:, col_off:col_off+7] = U_core
        
        # V: B row index = j*p + k. For pair t, columns 2t, 2t+1.
        # V_core row 0 (e): j=0, local_k=0 -> full row 0*p + 2t = 2t
        # V_core row 1 (f): j=0, local_k=1 -> full row 0*p + 2t+1 = 2t+1
        # V_core row 2 (g): j=1, local_k=0 -> full row 1*p + 2t = 16+2t
        # V_core row 3 (h): j=1, local_k=1 -> full row 1*p + 2t+1 = 16+2t+1
        V[2*t, col_off:col_off+7] = V_core[0, :]
        V[2*t+1, col_off:col_off+7] = V_core[1, :]
        V[16+2*t, col_off:col_off+7] = V_core[2, :]
        V[16+2*t+1, col_off:col_off+7] = V_core[3, :]
        
        # W: C row index = k*n + i. For pair t, output cols 2t, 2t+1.
        # W_core row 0 (C11): i=0, local_k=0 -> full row 2t*2 + 0 = 4t
        # W_core row 1 (C12): i=0, local_k=1 -> full row (2t+1)*2 + 0 = 4t+2
        # W_core row 2 (C21): i=1, local_k=0 -> full row 2t*2 + 1 = 4t+1
        # W_core row 3 (C22): i=1, local_k=1 -> full row (2t+1)*2 + 1 = 4t+3
        W[4*t, col_off:col_off+7] = W_core[0, :]
        W[4*t+2, col_off:col_off+7] = W_core[1, :]
        W[4*t+1, col_off:col_off+7] = W_core[2, :]
        W[4*t+3, col_off:col_off+7] = W_core[3, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
