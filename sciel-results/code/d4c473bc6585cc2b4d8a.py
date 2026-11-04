import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition
    # Indices for U2, V2, W2 are 0..3 (2x2 matrices)
    # Rank 7
    r2 = 7
    U2 = np.zeros((4, r2))
    V2 = np.zeros((4, r2))
    W2 = np.zeros((4, r2))
    
    # Strassen steps:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U2[i,j,t] is coeff of A_ij in L_t(A)
    # V2[j,k,t] is coeff of B_jk in R_t(B)
    # W2[k,i,t] is coeff in C_ik = sum W2[k,i,t] L_t(A) R_t(B)
    
    # t=0: M1
    # A: 00, 11. B: 00, 11. C: 00, 11 (with signs)
    # Actually, we just need the decomposition of the multiplication tensor.
    # The Strassen algorithm provides the decomposition.
    
    # Let's define the 7 linear forms for A and B, and the combination for C.
    # A forms:
    # L0 = A00 + A11
    # L1 = A10 + A11
    # L2 = A00
    # L3 = A11
    # L4 = A00 + A01
    # L5 = A10 - A00
    # L6 = A01 - A11
    
    # B forms:
    # R0 = B00 + B11
    # R1 = B00
    # R2 = B10 - B11
    # R3 = B01 - B00
    # R4 = B11
    # R5 = B00 + B01
    # R6 = B10 + B11
    
    # C combinations (for each t, C is a linear combo of M_t? No, C_ik is sum of W_ik_t M_t)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # So W2[k,i,t] is the coefficient of M_t in C_ik.
    
    # Initialize U2, V2, W2
    # U2[i,j,t]
    # t=0: A00(1), A11(1)
    U2[0, 0] = 1; U2[3, 0] = 1
    # t=1: A10(1), A11(1)
    U2[2, 1] = 1; U2[3, 1] = 1
    # t=2: A00(1)
    U2[0, 2] = 1
    # t=3: A11(1)
    U2[3, 3] = 1
    # t=4: A00(1), A01(1)
    U2[0, 4] = 1; U2[1, 4] = 1
    # t=5: A10(1), A00(-1)
    U2[2, 5] = 1; U2[0, 5] = -1
    # t=6: A01(1), A11(-1)
    U2[1, 6] = 1; U2[3, 6] = -1
    
    # V2[j,k,t]
    # t=0: B00(1), B11(1)
    V2[0, 0] = 1; V2[3, 0] = 1
    # t=1: B00(1)
    V2[0, 1] = 1
    # t=2: B10(1), B11(-1)
    V2[2, 2] = 1; V2[3, 2] = -1
    # t=3: B01(1), B00(-1)
    V2[1, 3] = 1; V2[0, 3] = -1
    # t=4: B11(1)
    V2[3, 4] = 1
    # t=5: B00(1), B01(1)
    V2[0, 5] = 1; V2[1, 5] = 1
    # t=6: B10(1), B11(1)
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # W2[k,i,t]
    # C00 (i=0,k=0): M1(1), M4(1), M5(-1), M7(-1)
    W2[0, 0, 0] = 1; W2[0, 0, 3] = 1; W2[0, 0, 4] = -1; W2[0, 0, 6] = -1
    # C01 (i=0,k=1): M3(1), M5(1)
    W2[1, 0, 2] = 1; W2[1, 0, 4] = 1
    # C10 (i=1,k=0): M2(1), M4(1)
    W2[0, 1, 1] = 1; W2[0, 1, 3] = 1
    # C11 (i=1,k=1): M1(1), M2(-1), M3(1), M7(1)
    W2[1, 1, 0] = 1; W2[1, 1, 1] = -1; W2[1, 1, 2] = 1; W2[1, 1, 6] = 1
    
    # Now compose to 4x4
    r4 = 49
    U4 = np.zeros((16, r4))
    V4 = np.zeros((16, r4))
    W4 = np.zeros((16, r4))
    
    # Strassen steps for 4x4 (blocks are 2x2)
    # We need to define X_k and Y_k in terms of 2x2 blocks.
    # Blocks of A: A00 (0-1,0-1), A01 (0-1,2-3), A10 (2-3,0-1), A11 (2-3,2-3)
    # Let's map global 4x4 indices (i,j) to local 2x2 block indices.
    # Block (bi, bj) where bi, bj in 0..1.
    # Local indices li, lj in 0..1.
    # Global i = bi*2 + li, j = bj*2 + lj.
    
    # Define the 7 X_k and Y_k linear forms on the blocks.
    # X_k is a 2x2 matrix formed by linear combo of A blocks.
    # Y_k is a 2x2 matrix formed by linear combo of B blocks.
    
    # X0 = A00 + A11
    # X1 = A10 + A11
    # X2 = A00
    # X3 = A11
    # X4 = A00 + A01
    # X5 = A10 - A00
    # X6 = A01 - A11
    
    # Y0 = B00 + B11
    # Y1 = B00
    # Y2 = B10 - B11
    # Y3 = B01 - B00
    # Y4 = B11
    # Y5 = B00 + B01
    # Y6 = B10 + B11
    
    # For each k in 0..6, and s in 0..6 (inner Strassen step):
    # t = k*7 + s
    # U4[global_i, global_j, t] = U2[li, lj, s] * coeff_Xk(block_i, block_j)
    # V4[global_j, global_k, t] = V2[lj, lk, s] * coeff_Yk(block_j, block_k)
    # W4[global_k, global_i, t] = W2[lk, li, s] * coeff_Ck(block_k, block_i)
    
    # We need coeff_Ck for the 4x4 output blocks.
    # C00 = M1 + M4 - M5 - M7  -> k=0: 1, k=3: 1, k=4: -1, k=6: -1
    # C01 = M3 + M5            -> k=2: 1, k=4: 1
    # C10 = M2 + M4            -> k=1: 1, k=3: 1
    # C11 = M1 - M2 + M3 + M7  -> k=0: 1, k=1: -1, k=2: 1, k=6: 1
    
    # Let's create helper functions to get coefficients.
    
    def get_X_coeff(k, bi, bj):
        if k == 0: return 1 if (bi==0 and bj==0) or (bi==1 and bj==1) else 0
        if k == 1: return 1 if (bi==1 and bj==0) or (bi==1 and bj==1) else 0
        if k == 2: return 1 if (bi==0 and bj==0) else 0
        if k == 3: return 1 if (bi==1 and bj==1) else 0
        if k == 4: return 1 if (bi==0 and bj==0) or (bi==0 and bj==1) else 0
        if k == 5: return 1 if (bi==1 and bj==0) else -1 if (bi==0 and bj==0) else 0
        if k == 6: return 1 if (bi==0 and bj==1) else -1 if (bi==1 and bj==1) else 0
        return 0
        
    def get_Y_coeff(k, bj, bk):
        if k == 0: return 1 if (bj==0 and bk==0) or (bj==1 and bk==1) else 0
        if k == 1: return 1 if (bj==0 and bk==0) else 0
        if k == 2: return 1 if (bj==1 and bk==0) else -1 if (bj==1 and bk==1) else 0
        if k == 3: return 1 if (bj==0 and bk==1) else -1 if (bj==0 and bk==0) else 0
        if k == 4: return 1 if (bj==1 and bk==1) else 0
        if k == 5: return 1 if (bj==0 and bk==0) or (bj==0 and bk==1) else 0
        if k == 6: return 1 if (bj==1 and bk==0) or (bj==1 and bk==1) else 0
        return 0
        
    def get_W_coeff(k, bk, bi):
        # C_bk,bi depends on M_k
        if bk == 0 and bi == 0:
            if k == 0: return 1
            if k == 3: return 1
            if k == 4: return -1
            if k == 6: return -1
            return 0
        if bk == 0 and bi == 1:
            if k == 2: return 1
            if k == 4: return 1
            return 0
        if bk == 1 and bi == 0:
            if k == 1: return 1
            if k == 3: return 1
            return 0
        if bk == 1 and bi == 1:
            if k == 0: return 1
            if k == 1: return -1
            if k == 2: return 1
            if k == 6: return 1
            return 0
        return 0

    for k in range(7):
        for s in range(7):
            t = k * 7 + s
            for bi in range(2):
                for bj in range(2):
                    cx = get_X_coeff(k, bi, bj)
                    if cx != 0:
                        for li in range(2):
                            for lj in range(2):
                                gi = bi * 2 + li
                                gj = bj * 2 + lj
                                U4[gi * 4 + gj, t] = U2[li * 2 + lj, s] * cx
                                
            for bj in range(2):
                for bk in range(2):
                    cy = get_Y_coeff(k, bj, bk)
                    if cy != 0:
                        for lj in range(2):
                            for lk in range(2):
                                gj = bj * 2 + lj
                                gk = bk * 2 + lk
                                V4[gj * 4 + gk, t] = V2[lj * 2 + lk, s] * cy
                                
            for bk in range(2):
                for bi in range(2):
                    cw = get_W_coeff(k, bk, bi)
                    if cw != 0:
                        for lk in range(2):
                            for li in range(2):
                                gk = bk * 2 + lk
                                gi = bi * 2 + li
                                W4[gk * 4 + gi, t] = W2[lk * 2 + li, s] * cw

    # Now extend to 4x8
    r = 98
    U = np.zeros((16, r))
    V = np.zeros((32, r))
    W = np.zeros((32, r))
    
    # U is same as U4
    U[:, :49] = U4
    
    # V: rows 0..15 for k=0..3, rows 16..31 for k=4..7
    # V4 has rows j*4+k.
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, :49] = V4[j * 4 + k, :]
            V[j * 8 + k + 4, :49] = V4[j * 4 + k, :]
            
    # W: rows k*4+i.
    # W4 has rows k*4+i.
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :49] = W4[k * 4 + i, :]
            W[(k + 4) * 4 + i, :49] = W4[k * 4 + i, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
