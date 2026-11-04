import numpy as np

def get_strassen_2x2():
    # U, V, W for 2x2 Strassen
    # A: [A11, A12, A21, A22]
    # B: [B11, B12, B21, B22]
    # C: [C11, C12, C21, C22]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C11 += M1, C22 += M1
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C21 += M2, C22 -= M2
    
    # M3 = A11(B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C12 += M3, C22 += M3
    
    # M4 = A22(B21 - B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C11 += M4, C21 += M4
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C11 -= M5, C12 += M5
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22 += M6
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C11 += M7
    
    return U, V, W

def construct():
    n, m, p = 5, 5, 5
    rank = 94
    
    # Get 2x2 Strassen
    U2, V2, W2 = get_strassen_2x2()
    
    # Construct 4x4 Strassen
    # A is 16 elements: [A11(4), A12(4), A21(4), A22(4)]
    # B is 16 elements: [B11(4), B12(4), B21(4), B22(4)]
    # C is 16 elements: [C11(4), C12(4), C21(4), C22(4)]
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Block indices for 4x4:
    # 0: A11, 1: A12, 2: A21, 3: A22
    # Each block has 4 elements.
    
    # X coefficients for 7 multiplications
    # X1 = A11 + A22
    # X2 = A21 + A22
    # X3 = A11
    # X4 = A22
    # X5 = A11 + A12
    # X6 = A21 - A11
    # X7 = A12 - A22
    
    # Y coefficients are similar for B
    
    # We iterate k=0..6
    for k in range(7):
        # Determine which blocks are involved and their signs
        # X_k is a 2x2 matrix (4 elements)
        # We need to map the 16 elements of A to the 4 elements of X_k
        # Then multiply by U2 (4x7) to get 7 scalar coefficients
        
        # Create a 16x4 matrix U_block_k
        U_block_k = np.zeros((16, 4))
        V_block_k = np.zeros((16, 4))
        
        # Helper to set block
        def set_block(matrix, block_idx, sign):
            start = block_idx * 4
            matrix[start:start+4, :] = sign * np.eye(4)
            
        # X coefficients
        if k == 0: # X1 = A11 + A22
            set_block(U_block_k, 0, 1)
            set_block(U_block_k, 3, 1)
        elif k == 1: # X2 = A21 + A22
            set_block(U_block_k, 2, 1)
            set_block(U_block_k, 3, 1)
        elif k == 2: # X3 = A11
            set_block(U_block_k, 0, 1)
        elif k == 3: # X4 = A22
            set_block(U_block_k, 3, 1)
        elif k == 4: # X5 = A11 + A12
            set_block(U_block_k, 0, 1)
            set_block(U_block_k, 1, 1)
        elif k == 5: # X6 = A21 - A11
            set_block(U_block_k, 2, 1)
            set_block(U_block_k, 0, -1)
        elif k == 6: # X7 = A12 - A22
            set_block(U_block_k, 1, 1)
            set_block(U_block_k, 3, -1)
            
        # Y coefficients (same structure)
        if k == 0:
            set_block(V_block_k, 0, 1)
            set_block(V_block_k, 3, 1)
        elif k == 1:
            set_block(V_block_k, 0, 1)
        elif k == 2:
            set_block(V_block_k, 1, 1)
            set_block(V_block_k, 3, -1)
        elif k == 3:
            set_block(V_block_k, 2, 1)
            set_block(V_block_k, 0, -1)
        elif k == 4:
            set_block(V_block_k, 3, 1)
        elif k == 5:
            set_block(V_block_k, 0, 1)
            set_block(V_block_k, 1, 1)
        elif k == 6:
            set_block(V_block_k, 2, 1)
            set_block(V_block_k, 3, 1)

        # Now U_block_k is 16x4. Multiply by U2 (4x7) to get 16x7
        U_scalar_k = U_block_k @ U2
        V_scalar_k = V_block_k @ V2
        
        # Assign to U4, V4
        start_col = k * 7
        U4[:, start_col:start_col+7] = U_scalar_k
        V4[:, start_col:start_col+7] = V_scalar_k
        
    # Construct W4
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M_k is 2x2. M_k = W2 @ m_k (where m_k is 7 scalars)
    # C_block = sum coeff_k M_k = sum coeff_k W2 m_k = W2 (sum coeff_k m_k)
    
    # So for each block of C, we sum the corresponding m_k vectors.
    
    # C11 (indices 0-3)
    # M1 (k=0), M4 (k=3), M5 (k=4), M7 (k=6)
    # Coeffs: 1, 1, -1, 1
    W4[0:4, 0:7] = W2 # M1
    W4[0:4, 21:28] = W2 # M4 (k=3, start=21)
    W4[0:4, 28:35] = -W2 # M5 (k=4, start=28)
    W4[0:4, 42:49] = W2 # M7 (k=6, start=42)
    
    # C12 (indices 4-7)
    # M3 (k=2), M5 (k=4)
    # Coeffs: 1, 1
    W4[4:8, 14:21] = W2 # M3 (k=2, start=14)
    W4[4:8, 28:35] = W2 # M5 (k=4, start=28)
    
    # C21 (indices 8-11)
    # M2 (k=1), M4 (k=3)
    # Coeffs: 1, 1
    W4[8:12, 7:14] = W2 # M2 (k=1, start=7)
    W4[8:12, 21:28] = W2 # M4 (k=3, start=21)
    
    # C22 (indices 12-15)
    # M1 (k=0), M2 (k=1), M3 (k=2), M6 (k=5)
    # Coeffs: 1, -1, 1, 1
    W4[12:16, 0:7] = W2 # M1
    W4[12:16, 7:14] = -W2 # M2
    W4[12:16, 14:21] = W2 # M3
    W4[12:16, 35:42] = W2 # M6 (k=5, start=35)
    
    # Now construct 5x5 U, V, W
    U = np.zeros((25, rank))
    V = np.zeros((25, rank))
    W = np.zeros((25, rank))
    
    # Fill 4x4 part (cols 0-48)
    U[0:16, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    col = 49
    
    # C41 = A44 B41 + A41 b55
    # A44: 0-15, B41: 16-19, C41: 16-19
    # A41: 16-19, B55: 24
    
    # A44 B41 (16 mults)
    for i in range(4):
        for j in range(4):
            U[i, col] = 1
            V[16+j, col] = 1
            W[16+i, col] = 1
            col += 1
            
    # A41 b55 (4 mults)
    for i in range(4):
        U[16+i, col] = 1
        V[24, col] = 1
        W[16+i, col] = 1
        col += 1
        
    # C14 = A14 B44 + a55 B14
    # A14: 20-23, B44: 0-15, C14: 20-23
    # A55: 24, B14: 20-23
    
    # A14 B44 (16 mults)
    for i in range(4):
        for j in range(4):
            U[20+i, col] = 1
            V[i*4+j, col] = 1
            W[20+j, col] = 1
            col += 1
            
    # a55 B14 (4 mults)
    for j in range(4):
        U[24, col] = 1
        V[20+j, col] = 1
        W[20+j, col] = 1
        col += 1
        
    # C55 = A41 B14 + a55 b55
    # A41: 16-19, B14: 20-23, C55: 24
    # A55: 24, B55: 24
    
    # A41 B14 (4 mults)
    for i in range(4):
        U[16+i, col] = 1
        V[20+i, col] = 1
        W[24, col] = 1
        col += 1
        
    # a55 b55 (1 mult)
    U[24, col] = 1
    V[24, col] = 1
    W[24, col] = 1
    col += 1
    
    assert col == 94
    
    return {"U": U, "V": V, "W": W}
