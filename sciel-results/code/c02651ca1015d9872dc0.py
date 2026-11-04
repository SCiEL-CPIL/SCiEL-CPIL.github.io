import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Strassen 2x2x2
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 1, -1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Helper to build 4x4x4 Strassen
    def build_444():
        # 7 blocks
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B12-B22)
        # M4: A22(B21-B11)
        # M5: (A11+A12)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        # Projections for A (16x4)
        # A11: rows 0,1 cols 0,1 -> flat 0,1,4,5
        # A12: rows 0,1 cols 2,3 -> flat 2,3,6,7
        # A21: rows 2,3 cols 0,1 -> flat 8,9,12,13
        # A22: rows 2,3 cols 2,3 -> flat 10,11,14,15
        
        def proj_A(block):
            P = np.zeros((16, 4))
            if block == 11:
                P[0,0]=1; P[1,1]=1; P[4,2]=1; P[5,3]=1
            elif block == 12:
                P[2,0]=1; P[3,1]=1; P[6,2]=1; P[7,3]=1
            elif block == 21:
                P[8,0]=1; P[9,1]=1; P[12,2]=1; P[13,3]=1
            elif block == 22:
                P[10,0]=1; P[11,1]=1; P[14,2]=1; P[15,3]=1
            return P
            
        def proj_B(block):
            P = np.zeros((16, 4))
            if block == 11:
                P[0,0]=1; P[1,1]=1; P[4,2]=1; P[5,3]=1
            elif block == 12:
                P[2,0]=1; P[3,1]=1; P[6,2]=1; P[7,3]=1
            elif block == 21:
                P[8,0]=1; P[9,1]=1; P[12,2]=1; P[13,3]=1
            elif block == 22:
                P[10,0]=1; P[11,1]=1; P[14,2]=1; P[15,3]=1
            return P
            
        def proj_C(block):
            P = np.zeros((16, 4))
            if block == 11:
                P[0,0]=1; P[1,1]=1; P[4,2]=1; P[5,3]=1
            elif block == 12:
                P[2,0]=1; P[3,1]=1; P[6,2]=1; P[7,3]=1
            elif block == 21:
                P[8,0]=1; P[9,1]=1; P[12,2]=1; P[13,3]=1
            elif block == 22:
                P[10,0]=1; P[11,1]=1; P[14,2]=1; P[15,3]=1
            return P

        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        # M1
        LA1 = proj_A(11) + proj_A(22)
        LB1 = proj_B(11) + proj_B(22)
        U4[:, 0:7] = LA1.T @ U2.T # Wait, LA1 is 16x4, U2 is 4x7. LA1.T is 4x16.
        # U_M = U2 @ L_A? No.
        # m = U2 @ (L_A @ A) = (U2 @ L_A) @ A.
        # U2 is 4x7. L_A is 16x4.
        # U2 @ L_A is not defined.
        # L_A maps A (16x1) to A' (4x1).
        # So L_A is 4x16.
        # U_M = U2 @ L_A. 4x7 @ 4x16 -> Error.
        # U2 is 4x7. L_A is 4x16.
        # We want U_M (16x7) such that U_M @ A = U2 @ (L_A @ A).
        # U_M @ A = U2 @ L_A @ A.
        # So U_M = U2 @ L_A.
        # But dimensions: U2 (4x7) L_A (4x16) is invalid.
        # L_A must be 16x4? No, L_A @ A (16x1) -> 4x1. So L_A is 4x16.
        # U2 (4x7) maps 4x1 to 7x1.
        # So U_M = U2 @ L_A is 4x7 @ 4x16. Invalid.
        # It should be L_A.T @ U2.T?
        # Let's check: (L_A.T @ U2.T) @ A = L_A.T @ (U2.T @ A).
        # U2.T is 7x4. A is 16x1. U2.T @ A is invalid.
        
        # Correct logic:
        # A' = L_A @ A. (4x1)
        # m = U2 @ A'. (7x1)
        # m = U2 @ L_A @ A.
        # So U_M = U2 @ L_A.
        # U2 is 4x7. L_A is 4x16.
        # This is still invalid.
        # Ah, U2 is defined such that m_i = sum U2[j,i] A'[j].
        # So m = U2.T @ A'?
        # If U2 is 4x7, and A' is 4x1, then U2.T @ A' is 7x1.
        # So m = U2.T @ L_A @ A.
        # U_M = U2.T @ L_A. (7x4 @ 4x16 = 7x16).
        # We want U_M to be 16x7.
        # So U_M = (U2.T @ L_A).T = L_A.T @ U2.
        # L_A.T is 16x4. U2 is 4x7.
        # Result 16x7. Correct.
        
        U4[:, 0:7] = LA1.T @ U2
        V4[:, 0:7] = LB1.T @ V2
        W4[:, 0:7] = proj_C(11).T @ W2 + proj_C(22).T @ W2 # No, W maps m to C.
        # C_block = W2 @ m.
        # C = L_C @ C_block.
        # C = L_C @ W2 @ m.
        # W_M = L_C @ W2.
        # L_C is 4x16 (maps C_block to C).
        # W2 is 4x7.
        # W_M = L_C @ W2. 4x16 @ 4x7 Invalid.
        # C_block (4x1) = W2.T @ m (7x1).
        # C (16x1) = L_C.T @ C_block?
        # L_C maps C (16x1) to C_block (4x1)? No, C_block is part of C.
        # C_block is 4x1. C is 16x1.
        # C_block = P_C @ C. (4x16 @ 16x1).
        # We want to add C_block to C.
        # C_new = C_old + P_C.T @ C_block.
        # So W_M adds to C.
        # W_M is 16x7.
        # C_block = W2.T @ m.
        # Contribution to C = P_C.T @ W2.T @ m.
        # W_M = P_C.T @ W2.T. (16x4 @ 7x4 -> Invalid).
        # W2 is 4x7. W2.T is 7x4.
        # P_C.T is 16x4.
        # P_C.T @ W2.T is 16x4 @ 7x4 Invalid.
        
        # Let's stick to: W_M = P_C.T @ W2.
        # P_C.T is 16x4. W2 is 4x7.
        # Result 16x7.
        # Check: W_M @ m = P_C.T @ W2 @ m.
        # W2 @ m is 4x1 (C_block).
        # P_C.T @ C_block is 16x1 (C).
        # Correct.
        
        W4[:, 0:7] = (proj_C(11) + proj_C(22)).T @ W2
        
        # M2
        LA2 = proj_A(21) + proj_A(22)
        LB2 = proj_B(11)
        U4[:, 7:14] = LA2.T @ U2
        V4[:, 7:14] = LB2.T @ V2
        W4[:, 7:14] = proj_C(21).T @ W2
        
        # M3
        LA3 = proj_A(11)
        LB3 = proj_B(12) - proj_B(22)
        U4[:, 14:21] = LA3.T @ U2
        V4[:, 14:21] = LB3.T @ V2
        W4[:, 14:21] = proj_C(12).T @ W2
        
        # M4
        LA4 = proj_A(22)
        LB4 = proj_B(21) - proj_B(11)
        U4[:, 21:28] = LA4.T @ U2
        V4[:, 21:28] = LB4.T @ V2
        W4[:, 21:28] = proj_C(22).T @ W2
        
        # M5
        LA5 = proj_A(11) + proj_A(12)
        LB5 = proj_B(22)
        U4[:, 28:35] = LA5.T @ U2
        V4[:, 28:35] = LB5.T @ V2
        W4[:, 28:35] = proj_C(22).T @ W2
        
        # M6
        LA6 = proj_A(21) - proj_A(11)
        LB6 = proj_B(11) + proj_B(12)
        U4[:, 35:42] = LA6.T @ U2
        V4[:, 35:42] = LB6.T @ V2
        W4[:, 35:42] = proj_C(22).T @ W2
        
        # M7
        LA7 = proj_A(12) - proj_A(22)
        LB7 = proj_B(21) + proj_B(22)
        U4[:, 42:49] = LA7.T @ U2
        V4[:, 42:49] = LB7.T @ V2
        W4[:, 42:49] = proj_C(22).T @ W2
        
        return U4, V4, W4

    U444, V444, W444 = build_444()
    
    # Schoolbook 4x4x3
    rank_443 = 4 * 4 * 3
    U443 = np.zeros((16, rank_443))
    V443 = np.zeros((12, rank_443)) # m*p = 4*3 = 12
    W443 = np.zeros((12, rank_443)) # p*n = 3*4 = 12
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U443[i * 4 + j, col] = 1
                V443[j * 3 + k, col] = 1
                W443[k * 4 + i, col] = 1
                col += 1
                
    # Assemble
    rank = 49 + rank_443
    U = np.zeros((16, rank))
    V = np.zeros((28, rank)) # 4*7
    W = np.zeros((28, rank)) # 7*4
    
    U[:, 0:49] = U444
    U[:, 49:] = U443
    
    # V444 uses B cols 0-3 (rows 0-15 of V)
    # V443 uses B cols 4-6 (rows 16-27 of V)
    V[0:16, 0:49] = V444
    V[16:28, 49:] = V443
    
    # W444 uses C cols 0-3 (rows 0-15 of W)
    # W443 uses C cols 4-6 (rows 16-27 of W)
    W[0:16, 0:49] = W444
    W[16:28, 49:] = W443
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
