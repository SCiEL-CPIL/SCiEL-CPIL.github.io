import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition for M_{2,3,3}
    # M_{2,3,3} = M_{2,2,2} (cols 0,1) + M_{2,2,1} (col 2) + M_{2,1,3} (all cols)
    
    # We will build U, V, W for M_{2,3,3} with rank 17
    # U: 6x17, V: 9x17, W: 6x17
    U_233 = np.zeros((6, 17))
    V_233 = np.zeros((9, 17))
    W_233 = np.zeros((6, 17))
    
    # 1. Strassen for M_{2,2,2} (A: 2x2, B: 2x2, C: 2x2)
    # A indices: 0,1,2,3 (i*m+j) -> 0:00, 1:01, 2:10, 3:11
    # B indices: 0,1,2,3,4,5,6,7,8 (j*p+k) for p=3? No, for M_{2,3,3}, m=3, p=3.
    # B is 3x3. Indices: 0:00, 1:01, 2:02, 3:10, 4:11, 5:12, 6:20, 7:21, 8:22.
    # C is 2x3. Indices: 0:00, 1:01, 2:02, 3:10, 4:11, 5:12.
    
    # Strassen terms for A_{0:2, 0:2} and B_{0:2, 0:2}
    # A00=0, A01=1, A10=2, A11=3
    # B00=0, B01=1, B10=3, B11=4
    # C00=0, C01=1, C10=3, C11=4
    
    # m1 = (a+d)(e+h)
    U_233[0, 0] = 1; U_233[3, 0] = 1
    V_233[0, 0] = 1; V_233[4, 0] = 1
    W_233[0, 0] = 1; W_233[4, 0] = 1 # C00, C11
    
    # m2 = (c+d)e
    U_233[2, 1] = 1; U_233[3, 1] = 1
    V_233[0, 1] = 1
    W_233[3, 1] = 1 # C10
    
    # m3 = a(f-h)
    U_233[0, 2] = 1
    V_233[1, 2] = 1; V_233[4, 2] = -1
    W_233[1, 2] = 1; W_233[4, 2] = 1 # C01, C11
    
    # m4 = d(g-c)
    U_233[3, 3] = 1
    V_233[3, 3] = 1; V_233[0, 3] = -1
    W_233[3, 3] = 1; W_233[0, 3] = -1 # C10, C00
    
    # m5 = (a+b)h
    U_233[0, 4] = 1; U_233[1, 4] = 1
    V_233[4, 4] = 1
    W_233[0, 4] = 1; W_233[1, 4] = 1 # C00, C01
    
    # m6 = (c-d)(f+g)
    U_233[2, 5] = 1; U_233[3, 5] = -1
    V_233[1, 5] = 1; V_233[3, 5] = 1
    W_233[1, 5] = 1; W_233[3, 5] = 1 # C01, C10
    
    # m7 = (b-c)(e+g)
    U_233[1, 6] = 1; U_233[2, 6] = -1
    V_233[0, 6] = 1; V_233[3, 6] = 1
    W_233[0, 6] = 1; W_233[3, 6] = -1 # C00, C10
    
    # 2. M_{2,2,1} for col 2 of B (indices 2, 5, 8)
    # A: 0,1,2,3. B: 2,5,8. C: 2,5.
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # t1 = A00*B02
    U_233[0, 7] = 1
    V_233[2, 7] = 1
    W_233[2, 7] = 1
    
    # t2 = A01*B12
    U_233[1, 8] = 1
    V_233[5, 8] = 1
    W_233[2, 8] = 1
    
    # t3 = A10*B02
    U_233[2, 9] = 1
    V_233[2, 9] = 1
    W_233[5, 9] = 1
    
    # t4 = A11*B12
    U_233[3, 10] = 1
    V_233[5, 10] = 1
    W_233[5, 10] = 1
    
    # 3. M_{2,1,3} for A_{2,1} (indices 4,5) and B_{1,3} (indices 6,7,8)
    # A: 4 (A02), 5 (A12). B: 6 (B20), 7 (B21), 8 (B22).
    # C: 0 (C00), 1 (C01), 2 (C02), 3 (C10), 4 (C11), 5 (C12).
    
    # u1 = A02*B20
    U_233[4, 11] = 1
    V_233[6, 11] = 1
    W_233[0, 11] = 1
    
    # u2 = A02*B21
    U_233[4, 12] = 1
    V_233[7, 12] = 1
    W_233[1, 12] = 1
    
    # u3 = A02*B22
    U_233[4, 13] = 1
    V_233[8, 13] = 1
    W_233[2, 13] = 1
    
    # u4 = A12*B20
    U_233[5, 14] = 1
    V_233[6, 14] = 1
    W_233[3, 14] = 1
    
    # u5 = A12*B21
    U_233[5, 15] = 1
    V_233[7, 15] = 1
    W_233[4, 15] = 1
    
    # u6 = A12*B22
    U_233[5, 16] = 1
    V_233[8, 16] = 1
    W_233[5, 16] = 1
    
    # Now tile for M_{2,3,9}
    # n=2, m=3, p=9
    # U: 6x51, V: 27x51, W: 18x51
    rank = 51
    U = np.zeros((6, rank))
    V = np.zeros((27, rank))
    W = np.zeros((18, rank))
    
    for block in range(3):
        col_offset = block * 17
        # U is shared
        U[:, col_offset:col_offset+17] = U_233
        
        # V: B is 3x9. Block l uses cols l*3 to l*3+2.
        # V_233 indices: 0..8 correspond to B_{0,0}..B_{2,2}.
        # For block l, B_{j, k} in 233 maps to B_{j, l*3+k} in 239.
        # Index in V_233: j*3+k.
        # Index in V_239: j*9 + (l*3+k).
        for j in range(3):
            for k in range(3):
                idx_233 = j*3 + k
                idx_239 = j*9 + (l*3 + k)
                V[idx_239, col_offset:col_offset+17] = V_233[idx_233]
                
        # W: C is 2x9. Block l uses cols l*3 to l*3+2.
        # W_233 indices: 0..5 correspond to C_{0,0}..C_{1,2}.
        # For block l, C_{i, k} in 233 maps to C_{i, l*3+k} in 239.
        # Index in W_233: i*3+k.
        # Index in W_239: i*9 + (l*3+k).
        for i in range(2):
            for k in range(3):
                idx_233 = i*3 + k
                idx_239 = i*9 + (l*3 + k)
                W[idx_239, col_offset:col_offset+17] = W_233[idx_233]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
