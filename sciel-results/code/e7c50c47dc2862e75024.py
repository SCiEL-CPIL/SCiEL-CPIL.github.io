import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create 2x2x2 Strassen decomposition
    def strassen_2x2():
        # U: 4x7, V: 4x7, W: 4x7
        # Indices for 2x2 matrix: r*2 + c
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = X00 * (Y00 + Y11)
        U[0, 0] = 1
        V[0, 0] = 1
        V[3, 0] = 1
        W[0, 0] = 1
        W[1, 0] = 1
        W[2, 0] = 1
        W[3, 0] = 1
        # Wait, W is for Z. Z = XY.
        # Z00 = M1 + M4 - M5 - M7
        # Z01 = M5 + M7
        # Z10 = M2 + M4 - M6 - M7
        # Z11 = M1 + M3 - M2 + M6
        
        # Let's fill U, V, W for each M
        
        # M1 = X00 * (Y00 + Y11)
        U[0, 0] = 1
        V[0, 0] = 1
        V[3, 0] = 1
        W[0, 0] = 1  # Z00
        W[3, 0] = 1  # Z11
        # Wait, Z00 is index 0, Z01 is 1, Z10 is 2, Z11 is 3.
        # M1 contributes to Z00 and Z11.
        # Z00 coeff 1, Z11 coeff 1.
        
        # M2 = (X00 + X10) * Y11
        U[0, 1] = 1
        U[2, 1] = 1
        V[3, 1] = 1
        W[2, 1] = 1  # Z10
        W[3, 1] = -1 # Z11
        
        # M3 = (X01 + X11) * Y00
        U[1, 2] = 1
        U[3, 2] = 1
        V[0, 2] = 1
        W[3, 2] = 1  # Z11
        
        # M4 = X11 * (Y01 + Y10)
        U[3, 3] = 1
        V[1, 3] = 1
        V[2, 3] = 1
        W[0, 3] = 1  # Z00
        W[2, 3] = 1  # Z10
        
        # M5 = X00 * (Y01 - Y11)
        U[0, 4] = 1
        V[1, 4] = 1
        V[3, 4] = -1
        W[0, 4] = -1 # Z00
        W[1, 4] = 1  # Z01
        
        # M6 = (X10 - X00) * (Y11 + Y10)
        U[2, 5] = 1
        U[0, 5] = -1
        V[3, 5] = 1
        V[2, 5] = 1
        W[2, 5] = -1 # Z10
        W[3, 5] = 1  # Z11
        
        # M7 = (X11 - X01) * (Y00 + Y01)
        U[3, 6] = 1
        U[1, 6] = -1
        V[0, 6] = 1
        V[1, 6] = 1
        W[0, 6] = -1 # Z00
        W[1, 6] = 1  # Z01
        W[2, 6] = -1 # Z10
        
        return U, V, W

    # Helper to create 4x4x4 decomposition using Strassen_2x2
    def strassen_4x4():
        U_blocks = []
        V_blocks = []
        W_blocks = []
        
        U_2, V_2, W_2 = strassen_2x2()
        
        # We need to map the 2x2 block indices to 4x4 indices.
        # A = [[A00, A01], [A10, A11]]
        # A00 is rows 0-1, cols 0-1.
        # A01 is rows 0-1, cols 2-3.
        # A10 is rows 2-3, cols 0-1.
        # A11 is rows 2-3, cols 2-3.
        
        # For each of the 7 M's, we need to create a 4x4 U, V, W
        # that corresponds to the multiplication of the specific blocks.
        
        for m_idx in range(7):
            U_m = np.zeros((16, 1)) # 4x4 matrix has 16 elements
            V_m = np.zeros((16, 1))
            W_m = np.zeros((16, 1))
            
            # Extract the coefficients for U, V, W for this M from U_2, V_2, W_2
            # U_2 has 4 rows (2x2 matrix).
            # We need to expand these to 16 rows (4x4 matrix).
            
            for r in range(4):
                for c in range(4):
                    # Determine which 2x2 block this (r,c) belongs to
                    # r_block = r // 2, c_block = c % 2
                    rb = r // 2
                    cb = c % 2
                    local_idx = rb * 2 + cb
                    
                    # Get the coefficient from U_2
                    coeff = U_2[local_idx, m_idx]
                    if coeff != 0:
                        U_m[r * 4 + c, 0] = coeff
            
            # Same for V
            # V is for B (4x4).
            # B = [[B00, B01], [B10, B11]]
            # B00 is rows 0-1, cols 0-1.
            # Note: In matrix mult C = AB, B's rows correspond to A's cols.
            # So B's index is (j, k). j is row of B, k is col of B.
            # In our 4x4 case, B is 4x4.
            # The 2x2 blocks of B are:
            # B00: rows 0-1, cols 0-1.
            # B01: rows 0-1, cols 2-3.
            # B10: rows 2-3, cols 0-1.
            # B11: rows 2-3, cols 2-3.
            
            for r in range(4):
                for c in range(4):
                    rb = r // 2
                    cb = c % 2
                    local_idx = rb * 2 + cb
                    coeff = V_2[local_idx, m_idx]
                    if coeff != 0:
                        V_m[r * 4 + c, 0] = coeff
            
            # Same for W
            # W is for C (4x4).
            # C = [[C00, C01], [C10, C11]]
            # C00: rows 0-1, cols 0-1.
            
            for r in range(4):
                for c in range(4):
                    rb = r // 2
                    cb = c % 2
                    local_idx = rb * 2 + cb
                    coeff = W_2[local_idx, m_idx]
                    if coeff != 0:
                        W_m[r * 4 + c, 0] = coeff
            
            U_blocks.append(U_m)
            V_blocks.append(V_m)
            W_blocks.append(W_m)
            
        U_4 = np.hstack(U_blocks)
        V_4 = np.hstack(V_blocks)
        W_4 = np.hstack(W_blocks)
        
        return U_4, V_4, W_4

    # Helper for 4x4x1 (schoolbook)
    def schoolbook_4x4x1():
        rank = 16
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        for i in range(4):
            for j in range(4):
                col = i * 4 + j
                U[i * 4 + j, col] = 1
                V[j * 4 + 0, col] = 1 # B is 4x1, so col 0
                W[0 * 4 + i, col] = 1 # C is 4x1, so col 0, row i
        return U, V, W

    # Combine for 4x4x9
    # Part 1: 4x4x4 (cols 0-3)
    U1, V1, W1 = strassen_4x4()
    # Map indices to global
    # U1 is 16x49. Global U is 16xR.
    # V1 is 16x49. Global V is 36xR (4*9).
    # W1 is 16x49. Global W is 36xR (9*4).
    
    # Part 2: 4x4x4 (cols 4-7)
    U2, V2, W2 = strassen_4x4()
    
    # Part 3: 4x4x1 (col 8)
    U3, V3, W3 = schoolbook_4x4x1()
    
    R = 49 + 49 + 16
    
    U_final = np.zeros((16, R))
    V_final = np.zeros((36, R))
    W_final = np.zeros((36, R))
    
    # Copy Part 1
    U_final[:, 0:49] = U1
    # V1 indices: local j*4+k. k is 0-3.
    # Global V index: j*9+k.
    for j in range(4):
        for k in range(4):
            V_final[j * 9 + k, 0:49] = V1[j * 4 + k, 0:49]
    # W1 indices: local k*4+i. k is 0-3.
    # Global W index: k*4+i.
    for k in range(4):
        for i in range(4):
            W_final[k * 4 + i, 0:49] = W1[k * 4 + i, 0:49]
            
    # Copy Part 2
    U_final[:, 49:98] = U2
    # V2 indices: local j*4+k. k is 0-3 (but these are global cols 4-7).
    # So local k=0 corresponds to global k=4.
    for j in range(4):
        for k in range(4):
            V_final[j * 9 + (k + 4), 49:98] = V2[j * 4 + k, 49:98] # Wait, V2 is 16x49.
            # V2[j*4+k, col]
    # W2 indices: local k*4+i. k is 0-3 (global 4-7).
    for k in range(4):
        for i in range(4):
            W_final[(k + 4) * 4 + i, 49:98] = W2[k * 4 + i, 49:98]
            
    # Copy Part 3
    U_final[:, 98:114] = U3
    # V3 indices: local j*4+0. Global col 8.
    for j in range(4):
        V_final[j * 9 + 8, 98:114] = V3[j * 4 + 0, 98:114]
    # W3 indices: local 0*4+i. Global col 8.
    for i in range(4):
        W_final[8 * 4 + i, 98:114] = W3[0 * 4 + i, 98:114]
        
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
