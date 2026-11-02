import numpy as np

DIMENSIONS = [2, 2, 16]

def _strassen_2x2():
    """
    Returns the Strassen decomposition for 2x2 matrix multiplication.
    U: (4, 7) - Coefficients for A (2x2 flattened to 4)
    V: (4, 7) - Coefficients for B (2x2 flattened to 4)
    W: (4, 7) - Coefficients for C (2x2 flattened to 4)
    
    Indices for 2x2 matrix [[a,b],[c,d]]:
    0: a, 1: b, 2: c, 3: d
    
    Strassen products:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    Output C:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    Uses Strassen's rank-7 decomposition for the 2x2 core, applied
    independently to each of the 8 column-pairs of the 2x16 matrix.
    Total rank = 7 * 8 = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    U_core, V_core, W_core = _strassen_2x2()
    
    num_pairs = p // 2  # 8 pairs
    rank = 7 * num_pairs  # 56
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for t in range(num_pairs):
        col_offset = t * 7
        
        for r in range(7):
            # U is the same for all pairs (A is 2x2)
            U[:, col_offset + r] = U_core[:, r]
            
            # V row index: j*p + k
            # j is row of B (0 or 1), k is column of B (0..15)
            # Pair t covers k = 2t, 2t+1
            # V_core has 4 rows: e, f, g, h corresponding to B [[e,f],[g,h]]
            # e is at (0, 2t), f is at (0, 2t+1)
            # g is at (1, 2t), h is at (1, 2t+1)
            # V_core indices: 0:e, 1:f, 2:g, 3:h
            
            V[2*t, col_offset + r] = V_core[0, r]       # e
            V[2*t + 1, col_offset + r] = V_core[1, r]   # f
            V[16 + 2*t, col_offset + r] = V_core[2, r]  # g
            V[16 + 2*t + 1, col_offset + r] = V_core[3, r] # h
            
            # W row index: k*n + i
            # k is column of C (0..15), i is row of C (0 or 1)
            # Pair t covers k = 2t, 2t+1
            # W_core has 4 rows: C11, C12, C21, C22
            # C11 is at (2t, 0) -> index 4*t + 0
            # C12 is at (2t+1, 0) -> index 4*(2t+1) + 0 = 8t + 4?
            # Wait, W shape is (p*n, rank) = (16*2, rank) = (32, rank).
            # Index is k*n + i.
            # For k=2t, i=0: 2t*2 + 0 = 4t.
            # For k=2t, i=1: 2t*2 + 1 = 4t+1.
            # For k=2t+1, i=0: (2t+1)*2 + 0 = 4t+2.
            # For k=2t+1, i=1: (2t+1)*2 + 1 = 4t+3.
            
            # W_core indices:
            # 0: C11 (i=0, k=0 in local 2x2)
            # 1: C12 (i=0, k=1 in local 2x2)
            # 2: C21 (i=1, k=0 in local 2x2)
            # 3: C22 (i=1, k=1 in local 2x2)
            
            # Mapping:
            # Local k=0 (even col), i=0 -> Global k=2t, i=0 -> W[4t]
            # Local k=1 (odd col), i=0 -> Global k=2t+1, i=0 -> W[4t+2]
            # Local k=0 (even col), i=1 -> Global k=2t, i=1 -> W[4t+1]
            # Local k=1 (odd col), i=1 -> Global k=2t+1, i=1 -> W[4t+3]
            
            W[4*t, col_offset + r] = W_core[0, r]       # C11
            W[4*t + 2, col_offset + r] = W_core[1, r]   # C12
            W[4*t + 1, col_offset + r] = W_core[2, r]   # C21
            W[4*t + 3, col_offset + r] = W_core[3, r]   # C22
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
