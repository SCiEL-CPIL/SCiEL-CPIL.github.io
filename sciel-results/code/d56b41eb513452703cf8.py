import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Define Strassen coefficients for 2x2 multiplication
    # P1 = x11(y11+y22)
    # P2 = (x11+x22)y22
    # P3 = (x12-x22)(y21+y22)
    # P4 = x11(y12-y22)
    # P5 = (x11+x12)y11
    # P6 = (x21-x11)(y11+y12)
    # P7 = (x12+x21)(y22-y11)
    
    # Coefficients for u (X) and v (Y) for each P_s
    # x indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # y indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U_strassen = np.zeros((7, 4))
    V_strassen = np.zeros((7, 4))
    
    # P1
    U_strassen[0, 0] = 1
    V_strassen[0, 0] = 1
    V_strassen[0, 3] = 1
    # P2
    U_strassen[1, 0] = 1
    U_strassen[1, 3] = 1
    V_strassen[1, 3] = 1
    # P3
    U_strassen[2, 1] = 1
    U_strassen[2, 3] = -1
    V_strassen[2, 2] = 1
    V_strassen[2, 3] = 1
    # P4
    U_strassen[3, 0] = 1
    V_strassen[3, 1] = 1
    V_strassen[3, 3] = -1
    # P5
    U_strassen[4, 0] = 1
    U_strassen[4, 1] = 1
    V_strassen[4, 0] = 1
    # P6
    U_strassen[5, 2] = 1
    U_strassen[5, 0] = -1
    V_strassen[5, 0] = 1
    V_strassen[5, 1] = 1
    # P7
    U_strassen[6, 1] = 1
    U_strassen[6, 2] = 1
    V_strassen[6, 3] = 1
    V_strassen[6, 0] = -1
    
    # Coefficients for Z (output of 2x2 mult) from P_s
    # Z11 = P1 + P4 - P5 + P7
    # Z12 = P3 + P5
    # Z21 = P2 + P4
    # Z22 = P1 - P2 + P3 + P6
    Z_strassen = np.zeros((7, 4)) # Rows are P_s, cols are Z entries 00,01,10,11
    Z_strassen[0, 0] = 1 # P1 -> Z11
    Z_strassen[0, 3] = 1 # P1 -> Z22
    Z_strassen[1, 2] = 1 # P2 -> Z21
    Z_strassen[1, 3] = -1 # P2 -> Z22
    Z_strassen[2, 1] = 1 # P3 -> Z12
    Z_strassen[2, 3] = 1 # P3 -> Z22
    Z_strassen[3, 0] = 1 # P4 -> Z11
    Z_strassen[3, 2] = 1 # P4 -> Z21
    Z_strassen[4, 0] = -1 # P5 -> Z11
    Z_strassen[4, 1] = 1 # P5 -> Z12
    Z_strassen[5, 3] = 1 # P6 -> Z22
    Z_strassen[6, 0] = 1 # P7 -> Z11
    
    # Outer Strassen steps for 4x4
    # Each step r defines X_r and Y_r as 2x2 matrices of blocks
    # We need to map block indices to 4x4 matrix indices
    # Block (br, bc) covers rows 2br..2br+1, cols 2bc..2bc+1
    
    def get_block_indices(br, bc):
        return [2*br + i for i in range(2)] * 2 + [2*bc + j for j in range(2)] # Not quite, need 4 pairs
        # Better: return list of (row, col) for the 4 elements
        # (2br, 2bc), (2br, 2bc+1), (2br+1, 2bc), (2br+1, 2bc+1)
    
    # Let's define the linear combinations for X_r and Y_r
    # X_r is 2x2, each entry is a sum of A blocks
    # A blocks: A00, A01, A10, A11
    
    # r=1: X=A00, Y=B00+B11
    # r=2: X=A00+A11, Y=B11
    # r=3: X=A01-A11, Y=B10+B11
    # r=4: X=A00, Y=B01-B11
    # r=5: X=A00+A01, Y=B00
    # r=6: X=A10-A00, Y=B00+B01
    # r=7: X=A01+A10, Y=B11-B00
    
    # We will create a list of 7 outer steps
    # Each step has:
    # - X_def: 2x2 array of coefficients for A blocks (4x4 matrix effectively)
    # - Y_def: 2x2 array of coefficients for B blocks
    # - C_def: 2x2 array of coefficients for C blocks (where M_r goes)
    
    # C reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    C_outer = np.zeros((7, 4)) # Rows M_r, cols C blocks 00,01,10,11
    C_outer[0, 0] = 1 # M1 -> C00
    C_outer[0, 3] = 1 # M1 -> C11
    C_outer[1, 2] = 1 # M2 -> C10
    C_outer[1, 3] = -1 # M2 -> C11
    C_outer[2, 1] = 1 # M3 -> C01
    C_outer[2, 3] = 1 # M3 -> C11
    C_outer[3, 0] = 1 # M4 -> C00
    C_outer[3, 2] = 1 # M4 -> C10
    C_outer[4, 0] = -1 # M5 -> C00
    C_outer[4, 1] = 1 # M5 -> C01
    C_outer[5, 3] = 1 # M6 -> C11
    C_outer[6, 0] = 1 # M7 -> C00
    
    # X_outer[r][i][j] is the coefficient of A_block(i,j) in X_r[i,j]
    # A_block(i,j) corresponds to rows 2i..2i+1, cols 2j..2j+1
    X_outer = np.zeros((7, 2, 2, 4, 4))
    Y_outer = np.zeros((7, 2, 2, 4, 4))
    
    # Helper to set a block in the 4x4 coeff matrix
    def set_block(coeff_mat, br, bc, val):
        coeff_mat[2*br:2*br+2, 2*bc:2*bc+2] += val
    
    # r=1
    set_block(X_outer[0, 0, 0], 0, 0, 1) # X1[0,0] = A00
    set_block(Y_outer[0, 0, 0], 0, 0, 1) # Y1[0,0] = B00
    set_block(Y_outer[0, 0, 0], 1, 1, 1) # Y1[0,0] += B11
    # Note: Y1[0,0] is the top-left block of Y1.
    # Wait, Y1 = B00 + B11.
    # B00 is top-left, B11 is bottom-right.
    # So Y1[0,0] (top-left of Y1) should be B00.
    # Y1[1,1] (bottom-right of Y1) should be B11.
    # My set_block adds to the 4x4 matrix.
    # X_outer[r, i, j] is the 4x4 matrix for X_r[i,j].
    # So X_outer[0, 0, 0] should have A00 in top-left.
    # Y_outer[0, 0, 0] should have B00 in top-left.
    # Y_outer[0, 1, 1] should have B11 in bottom-right?
    # No, Y1 is a 2x2 matrix of blocks.
    # Y1[0,0] = B00. Y1[0,1] = 0. Y1[1,0] = 0. Y1[1,1] = B11.
    # So Y_outer[0, 0, 0] gets B00.
    # Y_outer[0, 1, 1] gets B11.
    
    # Let's redo the outer definitions carefully.
    
    # r=1: X=A00, Y=B00+B11
    # X1[0,0]=A00, X1[0,1]=0, X1[1,0]=0, X1[1,1]=0
    # Y1[0,0]=B00, Y1[0,1]=0, Y1[1,0]=0, Y1[1,1]=B11
    set_block(X_outer[0, 0, 0], 0, 0, 1)
    set_block(Y_outer[0, 0, 0], 0, 0, 1)
    set_block(Y_outer[0, 1, 1], 1, 1, 1)
    
    # r=2: X=A00+A11, Y=B11
    # X2[0,0]=A00, X2[1,1]=A11
    # Y2[1,1]=B11
    set_block(X_outer[1, 0, 0], 0, 0, 1)
    set_block(X_outer[1, 1, 1], 1, 1, 1)
    set_block(Y_outer[1, 1, 1], 1, 1, 1)
    
    # r=3: X=A01-A11, Y=B10+B11
    # X3[0,1]=A01, X3[1,1]=-A11
    # Y3[1,0]=B10, Y3[1,1]=B11
    set_block(X_outer[2, 0, 1], 0, 1, 1)
    set_block(X_outer[2, 1, 1], 1, 1, -1)
    set_block(Y_outer[2, 1, 0], 1, 0, 1)
    set_block(Y_outer[2, 1, 1], 1, 1, 1)
    
    # r=4: X=A00, Y=B01-B11
    # X4[0,0]=A00
    # Y4[0,1]=B01, Y4[1,1]=-B11
    set_block(X_outer[3, 0, 0], 0, 0, 1)
    set_block(Y_outer[3, 0, 1], 0, 1, 1)
    set_block(Y_outer[3, 1, 1], 1, 1, -1)
    
    # r=5: X=A00+A01, Y=B00
    # X5[0,0]=A00, X5[0,1]=A01
    # Y5[0,0]=B00
    set_block(X_outer[4, 0, 0], 0, 0, 1)
    set_block(X_outer[4, 0, 1], 0, 1, 1)
    set_block(Y_outer[4, 0, 0], 0, 0, 1)
    
    # r=6: X=A10-A00, Y=B00+B01
    # X6[1,0]=A10, X6[0,0]=-A00
    # Y6[0,0]=B00, Y6[0,1]=B01
    set_block(X_outer[5, 1, 0], 1, 0, 1)
    set_block(X_outer[5, 0, 0], 0, 0, -1)
    set_block(Y_outer[5, 0, 0], 0, 0, 1)
    set_block(Y_outer[5, 0, 1], 0, 1, 1)
    
    # r=7: X=A01+A10, Y=B11-B00
    # X7[0,1]=A01, X7[1,0]=A10
    # Y7[1,1]=B11, Y7[0,0]=-B00
    set_block(X_outer[6, 0, 1], 0, 1, 1)
    set_block(X_outer[6, 1, 0], 1, 0, 1)
    set_block(Y_outer[6, 1, 1], 1, 1, 1)
    set_block(Y_outer[6, 0, 0], 0, 0, -1)
    
    # Now, generate the 49 terms for one 4x4x4 multiplication
    # We will create U_44, V_44, W_44 of shape (16, 49), (16, 49), (16, 49)
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    for r in range(7):
        for s in range(7):
            idx = r * 7 + s
            
            # Compute U coeff (16,)
            # U_s = sum_ij U_strassen[s, ij] * X_r[i,j]
            # X_r[i,j] is a 4x4 matrix (X_outer[r, i, j])
            u_coeff = np.zeros(16)
            for i in range(2):
                for j in range(2):
                    u_coeff += U_strassen[s, i*2+j] * X_outer[r, i, j].flatten()
            
            # Compute V coeff (16,)
            # V_s = sum_ij V_strassen[s, ij] * Y_r[i,j]
            v_coeff = np.zeros(16)
            for i in range(2):
                for j in range(2):
                    v_coeff += V_strassen[s, i*2+j] * Y_outer[r, i, j].flatten()
            
            # Compute W coeff (16,)
            # W_s = sum_mn Z_strassen[s, mn] * C_outer[r, mn]
            # Z_strassen[s, mn] is coeff of P_s in Z_r[mn]
            # C_outer[r, mn] is coeff of M_r in C_block(mn)
            # C_block(mn) is a 2x2 block of C.
            # We need to map C_block(mn) to 16 entries of C.
            w_coeff = np.zeros(16)
            for i in range(2):
                for j in range(2):
                    z_coeff = Z_strassen[s, i*2+j]
                    if z_coeff != 0:
                        # M_r contributes z_coeff * 1 to C_block(i,j)
                        # C_block(i,j) is at rows 2i..2i+1, cols 2j..2j+1
                        # But we also need to consider C_outer[r, i*2+j]
                        c_coeff = C_outer[r, i*2+j]
                        if c_coeff != 0:
                            val = z_coeff * c_coeff
                            # Add val to the 4 entries of C_block(i,j)
                            for di in range(2):
                                for dj in range(2):
                                    row = 2*i + di
                                    col = 2*j + dj
                                    w_coeff[row*4 + col] += val
            
            U_44[:, idx] = u_coeff
            V_44[:, idx] = v_coeff
            W_44[:, idx] = w_coeff
            
    # Now build the final U, V, W for 4x4x8
    # Part 1: B_left (cols 0-3)
    U[:, 0:49] = U_44
    # V_44 is 16x49, but V is 32x98.
    # V_44 rows correspond to B_left elements (16 of them).
    # In V (32x98), rows for B_left are 8j+k for j=0..3, k=0..3.
    # These are rows 0..3, 8..11, 16..19, 24..27.
    # V_44 row index is 4j+k.
    # So V_44[4j+k] goes to V[8j+k]
    for j in range(4):
        for k in range(4):
            V[8*j + k, 0:49] = V_44[4*j + k, :]
            
    # W_44 is 16x49. W is 32x98.
    # W_44 col index is 4i+k? No, W_44 is for C (4x4).
    # W_44 row index is 4i+k (where k is col of C, i is row of C).
    # In W (32x98), col index is 4k+i.
    # For B_left, k is 0..3.
    # So W_44[4i+k] goes to W[4k+i]
    for i in range(4):
        for k in range(4):
            W[4*k + i, 0:49] = W_44[4*i + k, :]
            
    # Part 2: B_right (cols 4-7)
    U[:, 49:98] = U_44
    # V_44 rows correspond to B_right elements.
    # B_right is B[:, 4:8].
    # In V (32x98), rows for B_right are 8j+k for j=0..3, k=4..7.
    # These are rows 4..7, 12..15, 20..23, 28..31.
    # V_44 row index is 4j+k' where k'=0..3.
    # So V_44[4j+k'] goes to V[8j + (k'+4)]
    for j in range(4):
        for k in range(4):
            V[8*j + (k+4), 49:98] = V_44[4*j + k, :]
            
    # W_44 cols correspond to C_right (cols 4-7).
    # In W (32x98), col index is 4k+i.
    # For B_right, k is 4..7.
    # So W_44[4i+k'] goes to W[4*(k'+4) + i]
    for i in range(4):
        for k in range(4):
            W[4*(k+4) + i, 49:98] = W_44[4*i + k, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

