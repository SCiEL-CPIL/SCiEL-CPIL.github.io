import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block indices for 4x4 matrix split into 2x2 blocks of 2x2
    # A_block(r, c) indices
    def get_A_block_indices(r, c):
        # r, c in {0, 1}
        # Rows 2r, 2r+1; Cols 2c, 2c+1
        return [
            (2*r) * 4 + (2*c),
            (2*r) * 4 + (2*c + 1),
            (2*r + 1) * 4 + (2*c),
            (2*r + 1) * 4 + (2*c + 1)
        ]
    
    def get_B_block_indices(r, c):
        # Same structure for B
        return [
            (2*r) * 4 + (2*c),
            (2*r) * 4 + (2*c + 1),
            (2*r + 1) * 4 + (2*c),
            (2*r + 1) * 4 + (2*c + 1)
        ]
    
    # Strassen steps definition
    # Each step defines X and Y as linear combinations of blocks
    # X is 2x2, Y is 2x2
    # We represent X and Y as lists of 4 vectors (length 16) for A and B
    # Actually, we just need the coefficients for the 16 elements.
    
    # Define the 7 steps
    # Step s: X = sum(alpha * A_block), Y = sum(beta * B_block)
    # We'll define the coefficients for X and Y elements directly.
    
    # Helper to create a zero 2x2 matrix of vectors
    def zero_2x2_vecs():
        return [[np.zeros(16) for _ in range(2)] for _ in range(2)]
    
    # Define the 7 Strassen steps for the 4x4 case
    # X and Y are 2x2 matrices of linear forms
    # We will build U and V columns for each of the 49 bilinear forms
    
    # Step definitions:
    # s=0: M1 = (A00+A11)(B00+B11)
    # s=1: M2 = (A10+A11)B00
    # s=2: M3 = A00(B01-B11)
    # s=3: M4 = A11(B10-B00)
    # s=4: M5 = (A00+A01)B11
    # s=5: M6 = (A10-A00)(B00+B01)
    # s=6: M7 = (A01-A11)(B10+B11)
    
    # For each step, we define the linear forms for X and Y
    # X is 2x2, so X[0][0], X[0][1], X[1][0], X[1][1]
    # Each is a vector of 16 coeffs
    
    steps = []
    
    # Step 0
    X = zero_2x2_vecs()
    Y = zero_2x2_vecs()
    idx_A00 = get_A_block_indices(0, 0)
    idx_A11 = get_A_block_indices(1, 1)
    idx_B00 = get_B_block_indices(0, 0)
    idx_B11 = get_B_block_indices(1, 1)
    
    for i in range(4):
        X[0][0][idx_A00[i]] += 1
        X[0][0][idx_A11[i]] += 1
        X[0][1][idx_A00[i+1 if i<2 else i-1]] += 1 # Wait, block indices order
        # Block indices: [r0c0, r0c1, r1c0, r1c1]
        # X[0][0] is top-left of X. Corresponds to A_block[0][0]
        # X[0][1] is top-right. Corresponds to A_block[0][1]
        # X[1][0] is bottom-left. Corresponds to A_block[1][0]
        # X[1][1] is bottom-right. Corresponds to A_block[1][1]
        
    # Let's redo the mapping carefully.
    # A_block(r, c) has elements:
    # (2r, 2c), (2r, 2c+1), (2r+1, 2c), (2r+1, 2c+1)
    # These correspond to local indices (0,0), (0,1), (1,0), (1,1) in the 2x2 block.
    
    # So for X = A00 + A11:
    # X[0][0] = A00[0][0] + A11[0][0]
    # X[0][1] = A00[0][1] + A11[0][1]
    # X[1][0] = A00[1][0] + A11[1][0]
    # X[1][1] = A00[1][1] + A11[1][1]
    
    def add_block_to_matrix(M, block_indices, coeff, block_r, block_c):
        # block_r, block_c is the local position in the 2x2 block (0 or 1)
        # M is 2x2 matrix of vectors
        # block_indices are the 4 global indices in order [00, 01, 10, 11]
        # We need to map local (i,j) to block index
        # Local (0,0) -> block index 0
        # Local (0,1) -> block index 1
        # Local (1,0) -> block index 2
        # Local (1,1) -> block index 3
        
        # Wait, get_A_block_indices returns [r0c0, r0c1, r1c0, r1c1]
        # So index 0 is (0,0), 1 is (0,1), 2 is (1,0), 3 is (1,1)
        
        # M[i][j] gets coeff * A_block[i][j]
        # A_block[i][j] is at global index block_indices[i*2 + j]
        
        for i in range(2):
            for j in range(2):
                global_idx = block_indices[i*2 + j]
                M[i][j][global_idx] += coeff

    # Step 0: M1
    X = zero_2x2_vecs()
    Y = zero_2x2_vecs()
    add_block_to_matrix(X, get_A_block_indices(0, 0), 1, 0, 0)
    add_block_to_matrix(X, get_A_block_indices(1, 1), 1, 0, 0) # Wait, A11 is block (1,1)
    # But X is the matrix being multiplied. X = A00 + A11.
    # A00 is block (0,0), A11 is block (1,1).
    # So X[0][0] = A00[0][0] + A11[0][0]
    # X[0][1] = A00[0][1] + A11[0][1]
    # etc.
    # So we add A00 to X and A11 to X.
    
    add_block_to_matrix(Y, get_B_block_indices(0, 0), 1, 0, 0)
    add_block_to_matrix(Y, get_B_block_indices(1, 1), 1, 0, 0)
    steps.append((X, Y))
    
    # Step 1: M2 = (A10 + A11)B00
    X = zero_2x2_vecs()
    Y = zero_2x2_vecs()
    add_block_to_matrix(X, get_A_block_indices(1, 0), 1, 0, 0)
    add_block_to_matrix(X, get_A_block_indices(1, 1), 1, 0, 0)
    add_block_to_matrix(Y, get_B_block_indices(0, 0), 1, 0, 0)
    steps.append((X, Y))
    
    # Step 2: M3 = A00(B01 - B11)
    X = zero_2x2_vecs()
    Y = zero_2x2_vecs()
    add_block_to_matrix(X, get_A_block_indices(0, 0), 1, 0, 0)
    add_block_to_matrix(Y, get_B_block_indices(0, 1), 1, 0, 0)
    add_block_to_matrix(Y, get_B_block_indices(1, 1), -1, 0, 0)
    steps.append((X, Y))
    
    # Step 3: M4 = A11(B10 - B00)
    X = zero_2x2_vecs()
    Y = zero_2x2_vecs()
    add_block_to_matrix(X, get_A_block_indices(1, 1), 1, 0, 0)
    add_block_to_matrix(Y, get_B_block_indices(1, 0), 1, 0, 0)
    add_block_to_matrix(Y, get_B_block_indices(0, 0), -1, 0, 0)
    steps.append((X, Y))
    
    # Step 4: M5 = (A00 + A01)B11
    X = zero_2x2_vecs()
    Y = zero_2x2_vecs()
    add_block_to_matrix(X, get_A_block_indices(0, 0), 1, 0, 0)
    add_block_to_matrix(X, get_A_block_indices(0, 1), 1, 0, 0)
    add_block_to_matrix(Y, get_B_block_indices(1, 1), 1, 0, 0)
    steps.append((X, Y))
    
    # Step 5: M6 = (A10 - A00)(B00 + B01)
    X = zero_2x2_vecs()
    Y = zero_2x2_vecs()
    add_block_to_matrix(X, get_A_block_indices(1, 0), 1, 0, 0)
    add_block_to_matrix(X, get_A_block_indices(0, 0), -1, 0, 0)
    add_block_to_matrix(Y, get_B_block_indices(0, 0), 1, 0, 0)
    add_block_to_matrix(Y, get_B_block_indices(0, 1), 1, 0, 0)
    steps.append((X, Y))
    
    # Step 6: M7 = (A01 - A11)(B10 + B11)
    X = zero_2x2_vecs()
    Y = zero_2x2_vecs()
    add_block_to_matrix(X, get_A_block_indices(0, 1), 1, 0, 0)
    add_block_to_matrix(X, get_A_block_indices(1, 1), -1, 0, 0)
    add_block_to_matrix(Y, get_B_block_indices(1, 0), 1, 0, 0)
    add_block_to_matrix(Y, get_B_block_indices(1, 1), 1, 0, 0)
    steps.append((X, Y))
    
    # Now generate U, V, W
    # For each step s (0..6) and each bilinear form p (0..6)
    
    # Bilinear forms for 2x2 multiplication
    # P0 = (X00+X11)(Y00+Y11)
    # P1 = (X10+X11)Y00
    # P2 = X00(Y01-Y11)
    # P3 = X11(Y10-Y00)
    # P4 = (X00+X01)Y11
    # P5 = (X10-X00)(Y00+Y01)
    # P6 = (X01-X11)(Y10+Y11)
    
    # Output Z = XY
    # Z00 = P0 + P3 - P4 + P6
    # Z01 = P2 + P4
    # Z10 = P1 + P3
    # Z11 = P0 - P1 + P2 + P5
    
    # W contributions for each step
    # M0 (P0..P6): C00 (+1), C11 (+1)
    # M1: C10 (+1), C11 (-1)
    # M2: C01 (+1), C11 (+1)
    # M3: C00 (+1), C10 (+1)
    # M4: C00 (-1), C01 (+1)
    # M5: C11 (+1)
    # M6: C00 (+1)
    
    # C blocks:
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    
    # W index: k*4 + i (col*4 + row)
    
    for s in range(7):
        X, Y = steps[s]
        for p in range(7):
            r = s * 7 + p
            
            # Determine U and V coefficients
            if p == 0:
                U_vec = np.array(X[0][0]) + np.array(X[1][1])
                V_vec = np.array(Y[0][0]) + np.array(Y[1][1])
            elif p == 1:
                U_vec = np.array(X[1][0]) + np.array(X[1][1])
                V_vec = np.array(Y[0][0])
            elif p == 2:
                U_vec = np.array(X[0][0])
                V_vec = np.array(Y[0][1]) - np.array(Y[1][1])
            elif p == 3:
                U_vec = np.array(X[1][1])
                V_vec = np.array(Y[1][0]) - np.array(Y[0][0])
            elif p == 4:
                U_vec = np.array(X[0][0]) + np.array(X[0][1])
                V_vec = np.array(Y[1][1])
            elif p == 5:
                U_vec = np.array(X[1][0]) - np.array(X[0][0])
                V_vec = np.array(Y[0][0]) + np.array(Y[0][1])
            elif p == 6:
                U_vec = np.array(X[0][1]) - np.array(X[1][1])
                V_vec = np.array(Y[1][0]) + np.array(Y[1][1])
            
            for i in range(16):
                U[i][r] = int(U_vec[i])
                V[i][r] = int(V_vec[i])
                
            # Determine W coefficients
            W_vec = np.zeros(16)
            
            # Z00 = P0 + P3 - P4 + P6
            # Z01 = P2 + P4
            # Z10 = P1 + P3
            # Z11 = P0 - P1 + P2 + P5
            
            z_coeffs = {}
            if p == 0:
                z_coeffs[(0,0)] = 1
                z_coeffs[(1,1)] = 1
            elif p == 1:
                z_coeffs[(1,0)] = 1
                z_coeffs[(1,1)] = -1
            elif p == 2:
                z_coeffs[(0,1)] = 1
                z_coeffs[(1,1)] = 1
            elif p == 3:
                z_coeffs[(0,0)] = 1
                z_coeffs[(1,0)] = 1
            elif p == 4:
                z_coeffs[(0,0)] = -1
                z_coeffs[(0,1)] = 1
            elif p == 5:
                z_coeffs[(1,1)] = 1
            elif p == 6:
                z_coeffs[(0,0)] = 1
            
            # Map Z[i][j] to C blocks
            # Z is the result of step s
            # Step 0 (M0): C00 (+1), C11 (+1)
            # Step 1 (M1): C10 (+1), C11 (-1)
            # Step 2 (M2): C01 (+1), C11 (+1)
            # Step 3 (M3): C00 (+1), C10 (+1)
            # Step 4 (M4): C00 (-1), C01 (+1)
            # Step 5 (M5): C11 (+1)
            # Step 6 (M6): C00 (+1)
            
            c_block_coeffs = {}
            if s == 0:
                c_block_coeffs[(0,0)] = 1
                c_block_coeffs[(1,1)] = 1
            elif s == 1:
                c_block_coeffs[(1,0)] = 1
                c_block_coeffs[(1,1)] = -1
            elif s == 2:
                c_block_coeffs[(0,1)] = 1
                c_block_coeffs[(1,1)] = 1
            elif s == 3:
                c_block_coeffs[(0,0)] = 1
                c_block_coeffs[(1,0)] = 1
            elif s == 4:
                c_block_coeffs[(0,0)] = -1
                c_block_coeffs[(0,1)] = 1
            elif s == 5:
                c_block_coeffs[(1,1)] = 1
            elif s == 6:
                c_block_coeffs[(0,0)] = 1
            
            # For each (zi, zj) in z_coeffs:
            # Z[zi][zj] contributes to C_block[zi][zj]
            # C_block(br, bc) is at rows 2br, 2br+1 and cols 2bc, 2bc+1
            # So Z[zi][zj] contributes to C[2br+zi][2bc+zj]
            
            for (zi, zj), zc in z_coeffs.items():
                # Find which C block this Z element belongs to
                # Z is the result of the 2x2 multiplication in step s
                # The step s defines which C blocks are affected
                # We need to map Z[zi][zj] to C block
                # Actually, Z is a 2x2 matrix.
                # The step s maps Z to C blocks.
                # For example, if s=0, Z is added to C00 and C11.
                # So Z[0][0] goes to C00[0][0] and C11[0][0].
                # Z[0][1] goes to C00[0][1] and C11[0][1].
                
                # We iterate over the C blocks that this step affects
                for (br, bc), cc in c_block_coeffs.items():
                    # Z[zi][zj] contributes to C[2br+zi][2bc+zj]
                    row = 2*br + zi
                    col = 2*bc + zj
                    w_idx = col * 4 + row
                    W_vec[w_idx] += zc * cc
            
            for i in range(16):
                W[i][r] = int(W_vec[i])
                
    return {"U": U, "V": V, "W": W}
