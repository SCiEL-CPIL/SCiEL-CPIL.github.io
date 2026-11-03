"""Matrix multiplication 2,4,8 with Strassen-based rectangular decomposition."""
DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We decompose the product A(2x4) * B(4x8) into 4 sub-products of type (2x2) * (2x4).
    # A is split into A_L (cols 0,1) and A_R (cols 2,3).
    # B is split into 4 blocks of (2x4):
    # B_TL (rows 0,1, cols 0,1,2,3)
    # B_TR (rows 0,1, cols 4,5,6,7)
    # B_BL (rows 2,3, cols 0,1,2,3)
    # B_BR (rows 2,3, cols 4,5,6,7)
    
    # Each (2x2) * (2x4) product can be decomposed into two (2x2) * (2x2) products.
    # Each (2x2) * (2x2) product uses Strassen's 7 rank-1 tensors.
    # So each (2x2) * (2x4) block uses 14 rank-1 tensors.
    # Total rank = 4 * 14 = 56.
    
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Helper to add Strassen tensors for a 2x2 * 2x2 block
    # a_indices: [i0, i1, j0, j1] in A (flattened index i*m + j)
    # b_indices: [k0, k1, l0, l1] in B (flattened index k*p + l)
    # c_indices: [i0, i1, l0, l1] in C (flattened index l*n + i)
    # col_offset: starting column index in U, V, W
    
    def add_strassen(a_idx, b_idx, c_idx, col_offset):
        # A elements:
        # a00 = A[a_idx[0]]
        # a01 = A[a_idx[1]]
        # a10 = A[a_idx[2]]
        # a11 = A[a_idx[3]]
        
        # B elements:
        # b00 = B[b_idx[0]]
        # b01 = B[b_idx[1]]
        # b10 = B[b_idx[2]]
        # b11 = B[b_idx[3]]
        
        # C elements:
        # c00 = C[c_idx[0]]
        # c01 = C[c_idx[1]]
        # c10 = C[c_idx[2]]
        # c11 = C[c_idx[3]]
        
        # Strassen products:
        # M1 = (a11 - a01) * (b00 + b10)
        # M2 = (a00 + a01) * b11
        # M3 = a00 * (b01 - b11)
        # M4 = a11 * (b10 - b00)
        # M5 = (a00 + a11) * (b00 + b11)
        # M6 = (a01 - a11) * (b10 + b11)
        # M7 = (a00 - a01) * (b00 + b01)
        
        # C00 = M5 + M4 - M1 - M7
        # C01 = M3 + M5
        # C10 = M1 + M2
        # C11 = M5 - M3 + M6 - M2
        
        # M1
        U[a_idx[3]][col_offset] += 1.0
        U[a_idx[1]][col_offset] -= 1.0
        V[b_idx[0]][col_offset] += 1.0
        V[b_idx[2]][col_offset] += 1.0
        W[c_idx[0]][col_offset] -= 1.0
        W[c_idx[2]][col_offset] += 1.0
        
        # M2
        U[a_idx[0]][col_offset + 1] += 1.0
        U[a_idx[1]][col_offset + 1] += 1.0
        V[b_idx[3]][col_offset + 1] += 1.0
        W[c_idx[2]][col_offset + 1] += 1.0
        W[c_idx[3]][col_offset + 1] -= 1.0
        
        # M3
        U[a_idx[0]][col_offset + 2] += 1.0
        V[b_idx[1]][col_offset + 2] += 1.0
        V[b_idx[3]][col_offset + 2] -= 1.0
        W[c_idx[1]][col_offset + 2] += 1.0
        W[c_idx[3]][col_offset + 2] -= 1.0
        
        # M4
        U[a_idx[3]][col_offset + 3] += 1.0
        V[b_idx[2]][col_offset + 3] += 1.0
        V[b_idx[0]][col_offset + 3] -= 1.0
        W[c_idx[0]][col_offset + 3] += 1.0
        
        # M5
        U[a_idx[0]][col_offset + 4] += 1.0
        U[a_idx[3]][col_offset + 4] += 1.0
        V[b_idx[0]][col_offset + 4] += 1.0
        V[b_idx[3]][col_offset + 4] += 1.0
        W[c_idx[0]][col_offset + 4] += 1.0
        W[c_idx[1]][col_offset + 4] += 1.0
        W[c_idx[3]][col_offset + 4] += 1.0
        
        # M6
        U[a_idx[1]][col_offset + 5] += 1.0
        U[a_idx[3]][col_offset + 5] -= 1.0
        V[b_idx[2]][col_offset + 5] += 1.0
        V[b_idx[3]][col_offset + 5] += 1.0
        W[c_idx[3]][col_offset + 5] += 1.0
        
        # M7
        U[a_idx[0]][col_offset + 6] += 1.0
        U[a_idx[1]][col_offset + 6] -= 1.0
        V[b_idx[0]][col_offset + 6] += 1.0
        V[b_idx[1]][col_offset + 6] += 1.0
        W[c_idx[0]][col_offset + 6] -= 1.0
    
    # Define indices for A (2x4)
    # A[i, j] -> index i * 4 + j
    # A_L: i=0,1; j=0,1
    # A_R: i=0,1; j=2,3
    
    # Define indices for B (4x8)
    # B[k, l] -> index k * 8 + l
    
    # Define indices for C (2x8)
    # C[i, l] -> index l * 2 + i  (Note: W[k*n+i,r] in problem statement? 
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Wait, the problem says W has shape (p*n, rank).
    # And Z[i,j] is usually row i, col j.
    # If W is indexed by k*n + i, that would be col-major for C?
    # Let's check the seed code:
    # W[k * n + i][col] = 1.0
    # Here k is the column of B (which is column of C), i is row of A (which is row of C).
    # So W index is col_C * n + row_C.
    # This means W is stored column-major for the C matrix?
    # C has shape n x p (2 x 8).
    # Index in W: col * n + row.
    # So for C[i, j], the index in W is j * n + i.
    
    # Let's re-verify the indices in add_strassen based on W index = col * n + row.
    # c_idx should be [C[0,0], C[0,1], C[1,0], C[1,1]] indices in W.
    # C[0,0] -> 0 * 2 + 0 = 0
    # C[0,1] -> 1 * 2 + 0 = 2
    # C[1,0] -> 0 * 2 + 1 = 1
    # C[1,1] -> 1 * 2 + 1 = 3
    
    # So c_idx = [0, 2, 1, 3] relative to the top-left of the 2x2 block.
    
    col_offset = 0
    
    # Block 1: A_L * B_TL
    # A_L: rows 0,1; cols 0,1.
    # a_idx: [A[0,0], A[0,1], A[1,0], A[1,1]] -> [0, 1, 4, 5]
    # B_TL: rows 0,1; cols 0,1,2,3.
    # We split B_TL into two 2x2 blocks: B_TL_0 (cols 0,1) and B_TL_1 (cols 2,3).
    
    # Sub-block 1.1: A_L * B_TL_0
    # b_idx for B_TL_0: [B[0,0], B[0,1], B[1,0], B[1,1]] -> [0, 1, 8, 9]
    # c_idx for C_TL_0: C[0,0], C[0,1], C[1,0], C[1,1]
    # C[0,0] idx 0, C[0,1] idx 2, C[1,0] idx 1, C[1,1] idx 3
    add_strassen([0, 1, 4, 5], [0, 1, 8, 9], [0, 2, 1, 3], col_offset)
    col_offset += 7
    
    # Sub-block 1.2: A_L * B_TL_1
    # b_idx for B_TL_1: [B[0,2], B[0,3], B[1,2], B[1,3]] -> [2, 3, 10, 11]
    # c_idx for C_TL_1: C[0,2], C[0,3], C[1,2], C[1,3]
    # C[0,2] idx 4, C[0,3] idx 6, C[1,2] idx 5, C[1,3] idx 7
    add_strassen([0, 1, 4, 5], [2, 3, 10, 11], [4, 6, 5, 7], col_offset)
    col_offset += 7
    
    # Block 2: A_L * B_TR
    # B_TR: rows 0,1; cols 4,5,6,7.
    # Split into B_TR_0 (cols 4,5) and B_TR_1 (cols 6,7).
    
    # Sub-block 2.1: A_L * B_TR_0
    # b_idx: [B[0,4], B[0,5], B[1,4], B[1,5]] -> [4, 5, 12, 13]
    # c_idx: C[0,4]..C[1,5]
    # C[0,4] idx 8, C[0,5] idx 10, C[1,4] idx 9, C[1,5] idx 11
    add_strassen([0, 1, 4, 5], [4, 5, 12, 13], [8, 10, 9, 11], col_offset)
    col_offset += 7
    
    # Sub-block 2.2: A_L * B_TR_1
    # b_idx: [B[0,6], B[0,7], B[1,6], B[1,7]] -> [6, 7, 14, 15]
    # c_idx: C[0,6]..C[1,7]
    # C[0,6] idx 12, C[0,7] idx 14, C[1,6] idx 13, C[1,7] idx 15
    add_strassen([0, 1, 4, 5], [6, 7, 14, 15], [12, 14, 13, 15], col_offset)
    col_offset += 7
    
    # Block 3: A_R * B_BL
    # A_R: rows 0,1; cols 2,3.
    # a_idx: [A[0,2], A[0,3], A[1,2], A[1,3]] -> [2, 3, 6, 7]
    # B_BL: rows 2,3; cols 0,1,2,3.
    # Split into B_BL_0 (cols 0,1) and B_BL_1 (cols 2,3).
    
    # Sub-block 3.1: A_R * B_BL_0
    # b_idx: [B[2,0], B[2,1], B[3,0], B[3,1]] -> [16, 17, 24, 25]
    # c_idx: C[0,0]..C[1,1] (same as Block 1.1)
    add_strassen([2, 3, 6, 7], [16, 17, 24, 25], [0, 2, 1, 3], col_offset)
    col_offset += 7
    
    # Sub-block 3.2: A_R * B_BL_1
    # b_idx: [B[2,2], B[2,3], B[3,2], B[3,3]] -> [18, 19, 26, 27]
    # c_idx: C[0,2]..C[1,3] (same as Block 1.2)
    add_strassen([2, 3, 6, 7], [18, 19, 26, 27], [4, 6, 5, 7], col_offset)
    col_offset += 7
    
    # Block 4: A_R * B_BR
    # B_BR: rows 2,3; cols 4,5,6,7.
    # Split into B_BR_0 (cols 4,5) and B_BR_1 (cols 6,7).
    
    # Sub-block 4.1: A_R * B_BR_0
    # b_idx: [B[2,4], B[2,5], B[3,4], B[3,5]] -> [20, 21, 28, 29]
    # c_idx: C[0,4]..C[1,5] (same as Block 2.1)
    add_strassen([2, 3, 6, 7], [20, 21, 28, 29], [8, 10, 9, 11], col_offset)
    col_offset += 7
    
    # Sub-block 4.2: A_R * B_BR_1
    # b_idx: [B[2,6], B[2,7], B[3,6], B[3,7]] -> [22, 23, 30, 31]
    # c_idx: C[0,6]..C[1,7] (same as Block 2.2)
    add_strassen([2, 3, 6, 7], [22, 23, 30, 31], [12, 14, 13, 15], col_offset)
    col_offset += 7
    
    return {"U": U, "V": V, "W": W}
