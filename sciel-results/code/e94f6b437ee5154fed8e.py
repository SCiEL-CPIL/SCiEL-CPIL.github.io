import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 32: 28 for top 2 rows (4 Strassen blocks) + 4 for bottom row
    rank = 32
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 factors
    # A indices: 0,1,2,3 -> a00, a01, a10, a11
    # B indices: 0,1,2,3 -> b00, b01, b10, b11
    # C indices: 0,1,2,3 -> c00, c01, c10, c11
    
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
    
    def add_strassen_block(col_offset, a_base, b_base, c_base):
        # a_base, b_base, c_base are lists of 4 global indices
        # a_base: [a00, a01, a10, a11]
        # b_base: [b00, b01, b10, b11]
        # c_base: [c00, c01, c10, c11]
        
        # M1
        U[a_base[0], col_offset] += 1
        U[a_base[3], col_offset] += 1
        V[b_base[0], col_offset] += 1
        V[b_base[3], col_offset] += 1
        W[c_base[0], col_offset] += 1
        W[c_base[3], col_offset] += 1
        
        # M2
        U[a_base[2], col_offset+1] += 1
        U[a_base[3], col_offset+1] += 1
        V[b_base[0], col_offset+1] += 1
        W[c_base[2], col_offset+1] += 1
        
        # M3
        U[a_base[0], col_offset+2] += 1
        V[b_base[1], col_offset+2] += 1
        V[b_base[3], col_offset+2] -= 1
        W[c_base[1], col_offset+2] += 1
        
        # M4
        U[a_base[3], col_offset+3] += 1
        V[b_base[2], col_offset+3] += 1
        V[b_base[0], col_offset+3] -= 1
        W[c_base[0], col_offset+3] += 1
        W[c_base[2], col_offset+3] += 1
        
        # M5
        U[a_base[0], col_offset+4] += 1
        U[a_base[1], col_offset+4] += 1
        V[b_base[3], col_offset+4] += 1
        W[c_base[0], col_offset+4] -= 1
        W[c_base[1], col_offset+4] += 1
        
        # M6
        U[a_base[2], col_offset+5] += 1
        U[a_base[0], col_offset+5] -= 1
        V[b_base[0], col_offset+5] += 1
        V[b_base[1], col_offset+5] += 1
        W[c_base[3], col_offset+5] += 1
        
        # M7
        U[a_base[1], col_offset+6] += 1
        U[a_base[3], col_offset+6] -= 1
        V[b_base[2], col_offset+6] += 1
        V[b_base[3], col_offset+6] += 1
        W[c_base[0], col_offset+6] += 1

    # Top 2 rows of A (indices 0-7 in U)
    # A is 3x4.
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    # A[2,0]=8, A[2,1]=9, A[2,2]=10, A[2,3]=11
    
    # B is 4x4.
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    # B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    # B[3,0]=12, B[3,1]=13, B[3,2]=14, B[3,3]=15
    
    # C is 3x4.
    # W index = k * n + i = k * 3 + i
    # C[0,0]=0, C[0,1]=3, C[0,2]=6, C[0,3]=9
    # C[1,0]=1, C[1,1]=4, C[1,2]=7, C[1,3]=10
    # C[2,0]=2, C[2,1]=5, C[2,2]=8, C[2,3]=11

    # Block 1: A_L (rows 0,1; cols 0,1) * B_TL (rows 0,1; cols 0,1) -> C_TL (rows 0,1; cols 0,1)
    # A_L indices: 0, 1, 4, 5
    # B_TL indices: 0, 1, 4, 5
    # C_TL indices: 0, 3, 1, 4
    add_strassen_block(0, [0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4])
    
    # Block 2: A_R (rows 0,1; cols 2,3) * B_BL (rows 2,3; cols 0,1) -> C_BL (rows 0,1; cols 0,1)
    # A_R indices: 2, 3, 6, 7
    # B_BL indices: 8, 9, 12, 13
    # C_BL indices: 0, 3, 1, 4
    add_strassen_block(7, [2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4])
    
    # Block 3: A_L * B_TR (rows 0,1; cols 2,3) -> C_TR (rows 0,1; cols 2,3)
    # A_L indices: 0, 1, 4, 5
    # B_TR indices: 2, 3, 6, 7
    # C_TR indices: 6, 9, 7, 10
    add_strassen_block(14, [0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10])
    
    # Block 4: A_R * B_BR (rows 2,3; cols 2,3) -> C_BR (rows 0,1; cols 2,3)
    # A_R indices: 2, 3, 6, 7
    # B_BR indices: 10, 11, 14, 15
    # C_BR indices: 6, 9, 7, 10
    add_strassen_block(21, [2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10])
    
    # Bottom row: A_bot (row 2) * B (all) -> C_bot (row 2)
    # Rank 4
    for j in range(4):
        r = 28 + j
        U[8 + j, r] = 1
        for k in range(4):
            V[j * 4 + k, r] = 1
            W[k * 3 + 2, r] = 1
            
    return {"U": U, "V": V, "W": W}
