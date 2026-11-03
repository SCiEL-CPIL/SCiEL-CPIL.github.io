import numpy as np

def get_strassen_2x2():
    # U, V, W for 2x2x2
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # 1: (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # 2: (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # 3: a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # 4: a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # 5: (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # 6: (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1
    
    # 7: (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def get_strassen_4x4():
    U2, V2, W2 = get_strassen_2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # 7 steps
    # Step 1: (A11+A22)(B11+B22)
    # A11: 0-3, A22: 12-15
    # B11: 0-3, B22: 12-15
    # C11: 0-3, C12: 4-7, C21: 8-11, C22: 12-15
    # M1 contributes to C11, C12, C21, C22?
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    # So M1 is not directly a block of C.
    # We need to use the 7 M's to reconstruct C.
    # C11 = M1 - M4 - M5 - M7 + M6? No.
    # Standard Strassen reconstruction:
    # C11 = M1 - M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6
    # Let's verify.
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # M4 = A22(B21-B11) = A22B21 - A22B11
    # M5 = (A11+A12)B22 = A11B22 + A12B22
    # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
    # M1 - M4 - M5 - M7 = (A11B11 + A11B22 + A22B11 + A22B22) - (A22B21 - A22B11) - (A11B22 + A12B22) - (A12B21 + A12B22 - A22B21 - A22B22)
    # = A11B11 + A11B22 + A22B11 + A22B22 - A22B21 + A22B11 - A11B22 - A12B22 - A12B21 - A12B22 + A22B21 + A22B22
    # = A11B11 + 2A22B11 - 2A12B22 + 2A22B22 - A12B21
    # This is not C11.
    # The reconstruction formulas are:
    # C11 = M1 - M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6
    # Wait, my M definitions might be slightly different from the standard ones.
    # Let's stick to the standard Strassen 7 products and reconstruction.
    
    # I will implement the 7 products as separate 2x2 multiplications and then combine them in W.
    
    steps = [
        # (A_indices, B_indices, C_indices, signs)
        # M1
        {
            'A': [(0,0,1), (3,3,1)], # A11 + A22
            'B': [(0,0,1), (3,3,1)], # B11 + B22
            'C': [(0,0,1), (0,1,1), (1,0,1), (1,1,1)], # All 4 blocks? No.
            # M1 is used in C11 and C22.
            # C11 = M1 - M4 - M5 - M7
            # C22 = M1 - M2 + M3 - M6
        }
    ]
    # This is getting complicated. I will just hardcode the 49 columns for U, V, W.
    # Or better, I will use the known 4x4x4 Strassen decomposition.
    
    # Actually, I can just use the 2x2x2 Strassen factors and embed them.
    # For each of the 7 M's, I create 7 columns in U, V, W.
    # For M1:
    # U part: A11 + A22.
    # V part: B11 + B22.
    # W part: M1 contributes to C11 and C22.
    # C11 = M1 - M4 - M5 - M7
    # C22 = M1 - M2 + M3 - M6
    # So for M1, W has +1 in C11 and +1 in C22.
    
    # Let's define the 7 M's and their contributions to C blocks.
    # M1: A11+A22, B11+B22 -> C11(+1), C22(+1)
    # M2: A21+A22, B11 -> C21(+1), C22(-1)
    # M3: A11, B12-B22 -> C12(+1), C22(+1)
    # M4: A22, B21-B11 -> C11(-1), C21(+1)
    # M5: A11+A12, B22 -> C11(-1), C12(+1)
    # M6: A21-A11, B11+B12 -> C22(-1)
    # M7: A12-A22, B21+B22 -> C11(-1)
    
    # I will build U, V, W column by column.
    
    def add_step(U, V, W, col_start, A_blocks, B_blocks, C_blocks):
        # A_blocks: list of (row_start, row_end, col_start, col_end, sign)
        # B_blocks: list of (row_start, row_end, col_start, col_end, sign)
        # C_blocks: list of (row_start, row_end, col_start, col_end, sign)
        for c in range(7):
            u_col = U[:, col_start + c]
            v_col = V[:, col_start + c]
            w_col = W[:, col_start + c]
            
            for (r1, r2, c1, c2, s) in A_blocks:
                # Map 2x2 U2 to this block
                # U2 is 4x7. Index 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                for i in range(2):
                    for j in range(2):
                        idx = i * 2 + j
                        u_idx = (r1 + i) * 4 + (c1 + j)
                        u_col[u_idx] += s * U2[idx, c]
                        
            for (r1, r2, c1, c2, s) in B_blocks:
                for i in range(2):
                    for j in range(2):
                        idx = i * 2 + j
                        v_idx = (r1 + i) * 4 + (c1 + j)
                        v_col[v_idx] += s * V2[idx, c]
                        
            for (r1, r2, c1, c2, s) in C_blocks:
                for i in range(2):
                    for j in range(2):
                        idx = i * 2 + j
                        w_idx = (r1 + i) * 4 + (c1 + j)
                        w_col[w_idx] += s * W2[idx, c]

    U2, V2, W2 = get_strassen_2x2()
    
    # M1
    add_step(U, V, W, 0, 
             [(0,0,1,1,1), (2,2,3,3,1)], # A11 + A22
             [(0,0,1,1,1), (2,2,3,3,1)], # B11 + B22
             [(0,0,1,1,1), (2,2,3,3,1)]) # C11 + C22
             
    # M2
    add_step(U, V, W, 7,
             [(2,0,3,3,1)], # A21 + A22? No, A21 is (2,0)-(3,1), A22 is (2,2)-(3,3)
             # A21: r 2-3, c 0-1. A22: r 2-3, c 2-3.
             # So A21+A22 is the whole bottom row of 2x2 blocks.
             # But Strassen M2 is (A21+A22)B11.
             # A21 is 2x2 block at (2,0). A22 is 2x2 block at (2,2).
             # So A_blocks: [(2,0,3,1,1), (2,2,3,3,1)]
             [(2,0,3,1,1), (2,2,3,3,1)],
             [(0,0,1,1,1)], # B11
             [(2,0,3,1,1), (2,2,3,3,-1)]) # C21 - C22?
             # C21 = M2 + M4. C22 = M1 - M2 + M3 - M6.
             # So M2 contributes +1 to C21 and -1 to C22.
             
    # M3
    add_step(U, V, W, 14,
             [(0,0,1,1,1)], # A11
             [(0,2,1,3,1), (2,2,3,3,-1)], # B12 - B22
             [(0,2,1,3,1), (2,2,3,3,1)]) # C12 + C22
             
    # M4
    add_step(U, V, W, 21,
             [(2,2,3,3,1)], # A22
             [(2,0,3,1,1), (0,0,1,1,-1)], # B21 - B11
             [(0,0,1,1,-1), (2,0,3,1,1)]) # C11 - C21
             
    # M5
    add_step(U, V, W, 28,
             [(0,0,1,1,1), (0,2,1,3,1)], # A11 + A12
             [(2,2,3,3,1)], # B22
             [(0,0,1,1,-1), (0,2,1,3,1)]) # C11 - C12
             
    # M6
    add_step(U, V, W, 35,
             [(2,0,3,1,1), (0,0,1,1,-1)], # A21 - A11
             [(0,0,1,1,1), (0,2,1,3,1)], # B11 + B12
             [(2,2,3,3,-1)]) # C22
             
    # M7
    add_step(U, V, W, 42,
             [(0,2,1,3,1), (2,2,3,3,-1)], # A12 - A22
             [(2,0,3,1,1), (2,2,3,3,1)], # B21 + B22
             [(0,0,1,1,-1)]) # C11
             
    return U, V, W

def construct():
    U4, V4, W4 = get_strassen_4x4()
    
    # For 4x4x8, we use two 4x4x4 multiplications.
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # First 4x4x4 (B1)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Second 4x4x4 (B2)
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
