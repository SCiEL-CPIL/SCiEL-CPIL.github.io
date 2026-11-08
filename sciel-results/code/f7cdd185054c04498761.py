import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    n, m, p = 3, 4, 4
    # Rank 32 construction
    # 2x4x4 part (rank 28) + 1x4x4 part (rank 4)
    
    U = np.zeros((n * m, 32), dtype=float)
    V = np.zeros((m * p, 32), dtype=float)
    W = np.zeros((p * n, 32), dtype=float)
    
    # Strassen factors for 2x2x2
    # U_s: 4x7, V_s: 4x7, W_s: 4x7
    U_s = np.zeros((4, 7), dtype=float)
    V_s = np.zeros((4, 7), dtype=float)
    W_s = np.zeros((4, 7), dtype=float)
    
    # Indices in 2x2 matrix: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # S1 = (a00+a11)(b00+b11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # S2 = (a10+a11)b00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    
    # S3 = a00(b10-b00)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # S4 = a11(b01-b11)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    # S5 = (a00+a01)b11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    # S6 = (a10-a00)(b00+b10)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # S7 = (a01-a11)(b01+b11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1
    
    # Helper to add a 2x2x2 block to global factors
    def add_block(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, rank_off):
        # a_r0, a_c0: top-left of A block (2x2)
        # b_r0, b_c0: top-left of B block (2x2)
        # c_r0, c_c0: top-left of C block (2x2) -- Note: C is (p,n) so (col, row)
        # c_r0 is row of C (0..2), c_c0 is col of C (0..3)
        
        for t in range(7):
            # U
            for r in range(2):
                for c in range(2):
                    li = r * 2 + c
                    gi = (a_r0 + r) * m + (a_c0 + c)
                    U[gi, rank_off + t] += U_s[li, t]
                    
            # V
            for r in range(2):
                for c in range(2):
                    li = r * 2 + c
                    gi = (b_r0 + r) * p + (b_c0 + c)
                    V[gi, rank_off + t] += V_s[li, t]
                    
            # W
            for r in range(2):
                for c in range(2):
                    li = r * 2 + c
                    # Global C index: (c_global, r_global)
                    # c_global = c_c0 + c
                    # r_global = c_r0 + r
                    gi = (c_c0 + c) * n + (c_r0 + r)
                    W[gi, rank_off + t] += W_s[li, t]
                    
    # 2x4x4 Part (Rows 0,1 of A)
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_block(0, 2, 2, 0, 0, 0, 7)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2]
    add_block(0, 0, 0, 2, 0, 2, 14)
    
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[2:4, 0:2]
    add_block(0, 2, 2, 2, 0, 2, 21)
    
    # 1x4x4 Part (Row 2 of A)
    # 4 terms
    for t in range(4):
        rank_idx = 28 + t
        # U: A[2, t]
        U[2 * m + t, rank_idx] = 1
        
        # V: B[t, t]
        V[t * p + t, rank_idx] = 1
        
        # W: C[t, 2]
        W[t * n + 2, rank_idx] = 1
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
