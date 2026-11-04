import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Try Rank 28 construction using Kronecker product
    # M_{2,4,6} = M_{2,2,2} \otimes M_{1,2,3}
    # Rank = 7 * 4 = 28
    
    rank_28 = 28
    U28 = np.zeros((nm, rank_28), dtype=float)
    V28 = np.zeros((mp, rank_28), dtype=float)
    W28 = np.zeros((pn, rank_28), dtype=float)
    
    # Strassen decomposition for 2x2x2
    # U_s, V_s, W_s are 2x2 matrices (flattened to 4x7)
    U_s = np.zeros((4, 7), dtype=float)
    V_s = np.zeros((4, 7), dtype=float)
    W_s = np.zeros((4, 7), dtype=float)
    
    # Term 0 (M1): (A00+A11)(B00+B11)
    U_s[0, 0] += 1.0
    U_s[3, 0] += 1.0
    V_s[0, 0] += 1.0
    V_s[3, 0] += 1.0
    W_s[0, 0] += 1.0
    W_s[3, 0] += 1.0
    
    # Term 1 (M2): (A10+A11)B00
    U_s[2, 1] += 1.0
    U_s[3, 1] += 1.0
    V_s[0, 1] += 1.0
    W_s[2, 1] += 1.0
    W_s[3, 1] -= 1.0
    
    # Term 2 (M3): A00(B01-B11)
    U_s[0, 2] += 1.0
    V_s[1, 2] += 1.0
    V_s[3, 2] -= 1.0
    W_s[1, 2] += 1.0
    W_s[3, 2] += 1.0
    
    # Term 3 (M4): A11(B10-B00)
    U_s[3, 3] += 1.0
    V_s[2, 3] += 1.0
    V_s[0, 3] -= 1.0
    W_s[0, 3] += 1.0
    W_s[2, 3] += 1.0
    
    # Term 4 (M5): (A00+A01)B11
    U_s[0, 4] += 1.0
    U_s[1, 4] += 1.0
    V_s[3, 4] += 1.0
    W_s[0, 4] -= 1.0
    W_s[1, 4] += 1.0
    
    # Term 5 (M6): (A10-A00)(B00+B01)
    U_s[2, 5] += 1.0
    U_s[0, 5] -= 1.0
    V_s[0, 5] += 1.0
    V_s[1, 5] += 1.0
    W_s[3, 5] += 1.0
    
    # Term 6 (M7): (A01-A11)(B10+B11)
    U_s[1, 6] += 1.0
    U_s[3, 6] -= 1.0
    V_s[2, 6] += 1.0
    V_s[3, 6] += 1.0
    W_s[0, 6] += 1.0

    # Rank 4 decomposition for 1x2x3
    # A' is 1x2, B' is 2x3, C' is 1x3
    # U4 is 2x4, V4 is 6x4, W4 is 3x4
    U4 = np.zeros((2, 4), dtype=float)
    V4 = np.zeros((6, 4), dtype=float)
    W4 = np.zeros((3, 4), dtype=float)
    
    # Hypothesized Rank 4 decomposition
    U4[0, 0] = 1.0
    V4[0, 0] = 1.0
    W4[0, 0] = 1.0
    
    U4[1, 1] = 1.0
    V4[3, 1] = 1.0
    W4[2, 1] = 1.0
    
    U4[0, 2] = 1.0
    U4[1, 2] = 1.0
    V4[1, 2] = 1.0
    V4[4, 2] = 1.0
    W4[1, 2] = 1.0
    
    U4[0, 3] = 1.0
    U4[1, 3] = -1.0
    V4[2, 3] = 1.0
    V4[5, 3] = 1.0
    W4[2, 3] = 1.0

    # Construct Kronecker Product
    # A_total (2x4) = A_s (2x2) \otimes A_4 (1x2)
    # Index mapping:
    # A_s index (i, j1) -> global i*2 + j1
    # A_4 index (j2) -> global j2
    # Total index: i*4 + j1*2 + j2
    
    # B_total (4x6) = B_s (2x2) \otimes B_4 (2x3)
    # B_s index (j1, k1) -> global j1*2 + k1
    # B_4 index (k2, l) -> global k2*3 + l
    # Total index: j1*6 + k1*3 + k2*3 + l? No.
    # B_s is 2x2. B_4 is 2x3.
    # B_total (i,j) where i in 0..3, j in 0..5.
    # i = j1*2 + k1? No.
    # B_s row j1, col k1. B_4 row k2, col l.
    # B_total row = j1*2 + k1? No, B_s is 2x2.
    # B_total = B_s \otimes B_4.
    # B_total[i, j] = B_s[i//2, i%2] * B_4[(i%2)*3 + j//3, j%3]?
    # Let's check dimensions.
    # B_s (2,2). B_4 (2,3).
    # B_total (4,6).
    # Row i = 2*j1 + j2? No.
    # Kronecker product of (2,2) and (2,3) is (4,6).
    # Row index r = 2*r_s + r_4.
    # Col index c = 2*c_s + c_4? No, col of B_4 is 3.
    # Col index c = 3*c_s + c_4.
    
    # W_total (2x6) = W_s (2x2) \otimes W_4 (1x3)
    # W_total (2,6).
    # Row index r = 2*r_s + r_4? No, W_4 is 1x3.
    # W_total row = r_s.
    # W_total col = 3*c_s + c_4.

    # Let's iterate over the 7 terms of Strassen and 4 terms of Rank 4
    for rs in range(7):
        for r4 in range(4):
            r_total = rs * 4 + r4
            
            # U (A)
            # A_s (4,7) -> indices 0..3
            # A_4 (2,4) -> indices 0..1
            # A_total (8, 28)
            # A_total index = 4 * i_s + i_4 ?
            # A_s is 2x2. A_4 is 1x2.
            # A_total = A_s \otimes A_4.
            # A_total (2, 4).
            # Row i = i_s.
            # Col j = 2*j_s + j_4.
            
            for i_s in range(2):
                for j_s in range(2):
                    for j_4 in range(2):
                        a_s_idx = i_s * 2 + j_s
                        a_4_idx = j_4
                        a_total_idx = i_s * 4 + j_s * 2 + j_4
                        
                        val_s = U_s[a_s_idx, rs]
                        val_4 = U4[a_4_idx, r4]
                        
                        if val_s != 0 and val_4 != 0:
                            U28[a_total_idx, r_total] += val_s * val_4

            # V (B)
            # B_s (4,7) -> indices 0..3
            # B_4 (6,4) -> indices 0..5
            # B_total (24, 28)
            # B_total = B_s \otimes B_4.
            # B_s (2,2). B_4 (2,3).
            # B_total (4,6).
            # Row r = 2*r_s + r_4? No.
            # B_s row r_s, col c_s.
            # B_4 row r_4, col c_4.
            # B_total row = 2*r_s + r_4? No.
            # B_total row = r_s * 2 + r_4?
            # B_total col = c_s * 3 + c_4.
            
            for r_s in range(2):
                for c_s in range(2):
                    for r_4 in range(2):
                        for c_4 in range(3):
                            b_s_idx = r_s * 2 + c_s
                            b_4_idx = r_4 * 3 + c_4
                            b_total_row = r_s * 2 + r_4
                            b_total_col = c_s * 3 + c_4
                            b_total_idx = b_total_row * 6 + b_total_col
                            
                            val_s = V_s[b_s_idx, rs]
                            val_4 = V4[b_4_idx, r4]
                            
                            if val_s != 0 and val_4 != 0:
                                V28[b_total_idx, r_total] += val_s * val_4

            # W (C)
            # W_s (4,7) -> indices 0..3
            # W_4 (3,4) -> indices 0..2
            # W_total (12, 28)
            # W_total = W_s \otimes W_4.
            # W_s (2,2). W_4 (1,3).
            # W_total (2,6).
            # Row r = r_s.
            # Col c = c_s * 3 + c_4.
            
            for r_s in range(2):
                for c_s in range(2):
                    for c_4 in range(3):
                        w_s_idx = r_s * 2 + c_s
                        w_4_idx = c_4
                        w_total_row = r_s
                        w_total_col = c_s * 3 + c_4
                        w_total_idx = w_total_row * 6 + w_total_col
                        
                        val_s = W_s[w_s_idx, rs]
                        val_4 = W4[w_4_idx, r4]
                        
                        if val_s != 0 and val_4 != 0:
                            W28[w_total_idx, r_total] += val_s * val_4

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank_28):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if verify(U28, V28, W28):
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    else:
        # Fallback to Rank 42
        return construct_rank_42()

def construct_rank_42():
    # ... existing code ...
    pass
