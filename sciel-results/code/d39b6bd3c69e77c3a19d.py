import numpy as np

def construct():
    n, m, p = 3, 4, 8
    # We will use a rank 64 decomposition based on Strassen for 2x2 blocks.
    # A1 (2x4) * B1 (4x4) -> C11 (2x4)
    # A1 (2x4) * B2 (4x4) -> C12 (2x4)
    # A2 (1x4) * B1 (4x4) -> C21 (1x4)
    # A2 (1x4) * B2 (4x4) -> C22 (1x4)
    
    # Strassen 2x2x2 has rank 7.
    # A1*B1 is 2x4 * 4x4. We can view it as 4 multiplications of 2x2 * 2x2.
    # A1 = [A11 A12], B1 = [B11; B21] (B1 is 4x4, so B11 is 2x2, B12 is 2x2, etc.)
    # Actually, A1 is 2x4, B1 is 4x4.
    # A1 = [[A11, A12]], where A11 is 2x2, A12 is 2x2.
    # B1 = [[B11, B12], [B21, B22]], where each is 2x2.
    # A1 * B1 = [A11*B11 + A12*B21, A11*B12 + A12*B22]
    # This is 4 multiplications of 2x2 by 2x2.
    # Each 2x2 by 2x2 multiplication has rank 7.
    # So A1*B1 has rank 4 * 7 = 28.
    
    # A2 is 1x4. B1 is 4x4.
    # A2 * B1 is 1x4.
    # This can be done with rank 4 (schoolbook).
    
    # Total rank = 28 + 28 + 4 + 4 = 64.
    
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen 2x2x2 factors
    # local_U_rows: list of 4 global U row indices
    # local_V_rows: list of 4 global V row indices
    # local_W_rows: list of 4 global W row indices
    # col_offset: starting column index
    def add_strassen_2x2(local_U, local_V, local_W, col_offset):
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # Note: In our 2x4 * 4x4 case, we have 4 separate 2x2 * 2x2 multiplications.
        # Let's define the 4 multiplications:
        # 1. A11 * B11 -> C11_part1
        # 2. A11 * B12 -> C12_part1
        # 3. A12 * B21 -> C11_part2
        # 4. A12 * B22 -> C12_part2
        
        # We need to map these to the Strassen formulas.
        # Actually, it's easier to just use the Strassen formulas for each 2x2 * 2x2 block.
        
        # Let's define the 4 blocks:
        # Block 1: A11 * B11
        # Block 2: A11 * B12
        # Block 3: A12 * B21
        # Block 4: A12 * B22
        
        # For each block, we add 7 columns.
        
        # Block 1: A11 * B11
        # A11 is local_U[0:2], B11 is local_V[0:2]
        # C11 is local_W[0:2]
        # We need to add the Strassen factors for A11 * B11.
        # But Strassen is for 2x2 * 2x2.
        # A11 is 2x2, B11 is 2x2.
        # So we can use the standard Strassen formulas.
        
        # Let's write a function to add Strassen factors for a single 2x2 * 2x2 multiplication.
        # local_U_2x2: 4 indices
        # local_V_2x2: 4 indices
        # local_W_2x2: 4 indices
        # col_offset: starting column
        def add_strassen_2x2_single(lU, lV, lW, col):
            # M1 = (A11 + A22)(B11 + B22)
            # A11 is lU[0], A22 is lU[3]
            # B11 is lV[0], B22 is lV[3]
            U[lU[0]][col] += 1
            U[lU[3]][col] += 1
            V[lV[0]][col] += 1
            V[lV[3]][col] += 1
            W[lW[0]][col] += 1
            W[lW[3]][col] += 1
            W[lW[1]][col] += 1
            W[lW[2]][col] += 1
            col += 1
            
            # M2 = (A21 + A22)B11
            # A21 is lU[2], A22 is lU[3]
            # B11 is lV[0]
            U[lU[2]][col] += 1
            U[lU[3]][col] += 1
            V[lV[0]][col] += 1
            W[lW[2]][col] += 1
            W[lW[3]][col] += 1
            col += 1
            
            # M3 = A11(B12 - B22)
            # A11 is lU[0]
            # B12 is lV[1], B22 is lV[3]
            U[lU[0]][col] += 1
            V[lV[1]][col] += 1
            V[lV[3]][col] -= 1
            W[lW[1]][col] += 1
            W[lW[3]][col] += 1
            col += 1
            
            # M4 = A22(B21 - B11)
            # A22 is lU[3]
            # B21 is lV[2], B11 is lV[0]
            U[lU[3]][col] += 1
            V[lV[2]][col] += 1
            V[lV[0]][col] -= 1
            W[lW[2]][col] += 1
            W[lW[3]][col] += 1
            col += 1
            
            # M5 = (A11 + A12)B22
            # A11 is lU[0], A12 is lU[1]
            # B22 is lV[3]
            U[lU[0]][col] += 1
            U[lU[1]][col] += 1
            V[lV[3]][col] += 1
            W[lW[1]][col] -= 1
            W[lW[3]][col] -= 1
            col += 1
            
            # M6 = (A21 - A11)(B11 + B12)
            # A21 is lU[2], A11 is lU[0]
            # B11 is lV[0], B12 is lV[1]
            U[lU[2]][col] += 1
            U[lU[0]][col] -= 1
            V[lV[0]][col] += 1
            V[lV[1]][col] += 1
            W[lW[3]][col] += 1
            col += 1
            
            # M7 = (A12 - A22)(B21 + B22)
            # A12 is lU[1], A22 is lU[3]
            # B21 is lV[2], B22 is lV[3]
            U[lU[1]][col] += 1
            U[lU[3]][col] -= 1
            V[lV[2]][col] += 1
            V[lV[3]][col] += 1
            W[lW[0]][col] -= 1
            W[lW[1]][col] += 1
            col += 1
            
            return col
        
        # Block 1: A11 * B11
        # A11 is local_U[0:2], B11 is local_V[0:2], C11 is local_W[0:2]
        # Wait, local_U is 4 indices for A11, A12, A21, A22.
        # A11 is local_U[0], local_U[1].
        # A12 is local_U[2], local_U[3].
        # B11 is local_V[0], local_V[1].
        # B12 is local_V[2], local_V[3].
        # B21 is local_V[4], local_V[5].
        # B22 is local_V[6], local_V[7].
        # C11 is local_W[0], local_W[1].
        # C12 is local_W[2], local_W[3].
        
        # So for Block 1 (A11 * B11):
        # lU = [local_U[0], local_U[1]]
        # lV = [local_V[0], local_V[1]]
        # lW = [local_W[0], local_W[1]]
        # But add_strassen_2x2_single expects 4 indices for each.
        # This is because Strassen is for 2x2 * 2x2.
        # A11 is 2x2, so it has 4 elements.
        # So lU should be the 4 global indices for A11.
        # lU = [local_U[0], local_U[1], local_U[2], local_U[3]]
        # But A11 is only 2x2, so it's 4 elements.
        # So lU = [local_U[0], local_U[1], local_U[2], local_U[3]] is correct for A11.
        # Wait, local_U is 4 indices for A11, A12, A21, A22.
        # So A11 is local_U[0], local_U[1].
        # A12 is local_U[2], local_U[3].
        # This is wrong. A11 is 2x2, so it has 4 elements.
        # So local_U should be 8 indices for A1 (2x4).
        # A11: local_U[0:4]
        # A12: local_U[4:8]
        
        # Let's redefine local_U, local_V, local_W to be the global indices for the 2x4, 4x4, 2x4 blocks.
        # local_U: 8 indices (2x4)
        # local_V: 16 indices (4x4)
        # local_W: 8 indices (2x4)
        
        # Block 1: A11 * B11
        # A11 is local_U[0:4]
        # B11 is local_V[0:4]
        # C11 is local_W[0:4]
        col_offset = add_strassen_2x2_single(local_U[0:4], local_V[0:4], local_W[0:4], col_offset)
        
        # Block 2: A11 * B12
        # A11 is local_U[0:4]
        # B12 is local_V[4:8]
        # C12 is local_W[4:8]
        col_offset = add_strassen_2x2_single(local_U[0:4], local_V[4:8], local_W[4:8], col_offset)
        
        # Block 3: A12 * B21
        # A12 is local_U[4:8]
        # B21 is local_V[8:12]
        # C11 is local_W[0:4]
        col_offset = add_strassen_2x2_single(local_U[4:8], local_V[8:12], local_W[0:4], col_offset)
        
        # Block 4: A12 * B22
        # A12 is local_U[4:8]
        # B22 is local_V[12:16]
        # C12 is local_W[4:8]
        col_offset = add_strassen_2x2_single(local_U[4:8], local_V[12:16], local_W[4:8], col_offset)
        
        return col_offset

    # A1 * B1
    # A1 is rows 0,1 of A. Cols 0,1,2,3.
    # U rows: 0,1,2,3,4,5,6,7
    # B1 is rows 0,1,2,3 of B. Cols 0,1,2,3.
    # V rows: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    # C11 is rows 0,1 of C. Cols 0,1,2,3.
    # W rows: 0,1,2,3,4,5,6,7
    local_U = [0, 1, 2, 3, 4, 5, 6, 7]
    local_V = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    local_W = [0, 1, 2, 3, 4, 5, 6, 7]
    col_offset = add_strassen_2x2(local_U, local_V, local_W, 0)
    
    # A1 * B2
    # A1 is rows 0,1 of A. Cols 0,1,2,3.
    # U rows: 0,1,2,3,4,5,6,7
    # B2 is rows 0,1,2,3 of B. Cols 4,5,6,7.
    # V rows: 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    # C12 is rows 0,1 of C. Cols 4,5,6,7.
    # W rows: 8,9,10,11,12,13,14,15
    local_U = [0, 1, 2, 3, 4, 5, 6, 7]
    local_V = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    local_W = [8, 9, 10, 11, 12, 13, 14, 15]
    col_offset = add_strassen_2x2(local_U, local_V, local_W, col_offset)
    
    # A2 * B1
    # A2 is row 2 of A. Cols 0,1,2,3.
    # U rows: 8,9,10,11
    # B1 is rows 0,1,2,3 of B. Cols 0,1,2,3.
    # V rows: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    # C21 is row 2 of C. Cols 0,1,2,3.
    # W rows: 16,17,18,19
    # Schoolbook: rank 4
    for j in range(4):
        U[8 + j][col_offset] = 1
        V[j][col_offset] = 1
        W[16 + j][col_offset] = 1
        col_offset += 1
    
    # A2 * B2
    # A2 is row 2 of A. Cols 0,1,2,3.
    # U rows: 8,9,10,11
    # B2 is rows 0,1,2,3 of B. Cols 4,5,6,7.
    # V rows: 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    # C22 is row 2 of C. Cols 4,5,6,7.
    # W rows: 20,21,22,23
    # Schoolbook: rank 4
    for j in range(4):
        U[8 + j][col_offset] = 1
        V[16 + j][col_offset] = 1
        W[20 + j][col_offset] = 1
        col_offset += 1
    
    return {"U": U, "V": V, "W": W}
