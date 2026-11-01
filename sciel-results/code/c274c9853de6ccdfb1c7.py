import numpy as np

def construct():
    n, m, p = 2, 4, 6
    
    # We will use a block decomposition.
    # A is 2x4, split into A1 (2x2) and A2 (2x2)
    # B is 4x6, split into B1 (2x6) and B2 (2x6)
    # C = A1 @ B1 + A2 @ B2
    # B1 is 2x6, split into B11, B12, B13 (each 2x2)
    # B2 is 2x6, split into B21, B22, B23 (each 2x2)
    # A1 @ B1 = A1 @ B11 + A1 @ B12 + A1 @ B13
    # A2 @ B2 = A2 @ B21 + A2 @ B22 + A2 @ B23
    # Each 2x2 @ 2x2 uses Strassen (rank 7)
    # Total rank = 3 * 7 + 3 * 7 = 42
    
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Indices for A: 0,1,2,3 (row-major 2x2)
    # Indices for B: 0,1,2,3 (row-major 2x2)
    # Indices for C: 0,1,2,3 (row-major 2x2)
    
    # Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the U, V, W contributions for one 2x2x2 block
    # The block corresponds to a specific A_sub (2x2) and B_sub (2x2)
    # and contributes to C_sub (2x2)
    
    # We need to map the local 2x2 indices to the global indices.
    # A is 2x4. 
    # A1 is rows 0-1, cols 0-1. Global indices: 0,1,2,3.
    # A2 is rows 0-1, cols 2-3. Global indices: 4,5,6,7.
    # B is 4x6.
    # B1 is rows 0-1, cols 0-5. Global indices: 0..5 (row 0), 6..11 (row 1).
    # B11 is rows 0-1, cols 0-1. Global indices: 0,1,6,7.
    # B12 is rows 0-1, cols 2-3. Global indices: 2,3,8,9.
    # B13 is rows 0-1, cols 4-5. Global indices: 4,5,10,11.
    # B2 is rows 2-3, cols 0-5. Global indices: 12..17 (row 2), 18..23 (row 3).
    # B21 is rows 2-3, cols 0-1. Global indices: 12,13,18,19.
    # B22 is rows 2-3, cols 2-3. Global indices: 14,15,20,21.
    # B23 is rows 2-3, cols 4-5. Global indices: 16,17,22,23.
    # C is 2x6.
    # C1 (from A1B1) is cols 0-5. Global indices: 0..5 (row 0), 6..11 (row 1).
    # C11 (from A1B11) is cols 0-1. Global indices: 0,1,6,7.
    # C12 (from A1B12) is cols 2-3. Global indices: 2,3,8,9.
    # C13 (from A1B13) is cols 4-5. Global indices: 4,5,10,11.
    # C2 (from A2B2) is cols 0-5. Global indices: 0..5 (row 0), 6..11 (row 1).
    # C21 (from A2B21) is cols 0-1. Global indices: 0,1,6,7.
    # C22 (from A2B22) is cols 2-3. Global indices: 2,3,8,9.
    # C23 (from A2B23) is cols 4-5. Global indices: 4,5,10,11.
    
    # Note: C1 and C2 add up to C. So W entries for C1 and C2 will be summed.
    
    def add_strassen_block(a_indices, b_indices, c_indices, start_col):
        # a_indices: list of 4 global U indices for the 2x2 A block
        # b_indices: list of 4 global V indices for the 2x2 B block
        # c_indices: list of 4 global W indices for the 2x2 C block
        # start_col: column index in U,V,W to start writing
        
        # M1
        U[a_indices[0], start_col] += 1
        U[a_indices[3], start_col] += 1
        V[b_indices[0], start_col] += 1
        V[b_indices[3], start_col] += 1
        W[c_indices[0], start_col] += 1
        W[c_indices[3], start_col] += 1
        
        # M2
        U[a_indices[2], start_col+1] += 1
        U[a_indices[3], start_col+1] += 1
        V[b_indices[0], start_col+1] += 1
        W[c_indices[2], start_col+1] += 1
        W[c_indices[3], start_col+1] += 1
        
        # M3
        U[a_indices[0], start_col+2] += 1
        V[b_indices[1], start_col+2] += 1
        V[b_indices[3], start_col+2] -= 1
        W[c_indices[1], start_col+2] += 1
        W[c_indices[3], start_col+2] += 1
        
        # M4
        U[a_indices[3], start_col+3] += 1
        V[b_indices[2], start_col+3] += 1
        V[b_indices[0], start_col+3] -= 1
        W[c_indices[0], start_col+3] += 1
        W[c_indices[2], start_col+3] += 1
        
        # M5
        U[a_indices[0], start_col+4] += 1
        U[a_indices[1], start_col+4] += 1
        V[b_indices[3], start_col+4] += 1
        W[c_indices[0], start_col+4] -= 1
        W[c_indices[1], start_col+4] += 1
        
        # M6
        U[a_indices[2], start_col+5] += 1
        U[a_indices[0], start_col+5] -= 1
        V[b_indices[0], start_col+5] += 1
        V[b_indices[1], start_col+5] += 1
        W[c_indices[3], start_col+5] += 1
        
        # M7
        U[a_indices[1], start_col+6] += 1
        U[a_indices[3], start_col+6] -= 1
        V[b_indices[2], start_col+6] += 1
        V[b_indices[3], start_col+6] += 1
        W[c_indices[0], start_col+6] += 1
        W[c_indices[2], start_col+6] -= 1
        
        return start_col + 7

    # A1 indices: 0,1,2,3
    # A2 indices: 4,5,6,7
    
    # B11 indices: 0,1,6,7
    # B12 indices: 2,3,8,9
    # B13 indices: 4,5,10,11
    
    # B21 indices: 12,13,18,19
    # B22 indices: 14,15,20,21
    # B23 indices: 16,17,22,23
    
    # C11 indices: 0,1,6,7
    # C12 indices: 2,3,8,9
    # C13 indices: 4,5,10,11
    
    # C21 indices: 0,1,6,7
    # C22 indices: 2,3,8,9
    # C23 indices: 4,5,10,11
    
    col = 0
    col = add_strassen_block([0,1,2,3], [0,1,6,7], [0,1,6,7], col) # A1 B11 -> C11
    col = add_strassen_block([0,1,2,3], [2,3,8,9], [2,3,8,9], col) # A1 B12 -> C12
    col = add_strassen_block([0,1,2,3], [4,5,10,11], [4,5,10,11], col) # A1 B13 -> C13
    
    col = add_strassen_block([4,5,6,7], [12,13,18,19], [0,1,6,7], col) # A2 B21 -> C21 (adds to C11)
    col = add_strassen_block([4,5,6,7], [14,15,20,21], [2,3,8,9], col) # A2 B22 -> C22 (adds to C12)
    col = add_strassen_block([4,5,6,7], [16,17,22,23], [4,5,10,11], col) # A2 B23 -> C23 (adds to C13)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
