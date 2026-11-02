def add_strassen_2x2(A_blocks, B_blocks, C_blocks, offset):
    # A_blocks: [A00_idx, A01_idx, A10_idx, A11_idx]
    # B_blocks: [B00_idx, B01_idx, B10_idx, B11_idx]
    # C_blocks: [C00_idx, C01_idx, C10_idx, C11_idx]
    
    # M1 = (A00+A11)(B00+B11)
    for i in range(4):
        U[A_blocks[0][i], offset] += 1
        U[A_blocks[3][i], offset] += 1
        V[B_blocks[0][i], offset] += 1
        V[B_blocks[3][i], offset] += 1
    W[C_blocks[0][0], offset] += 1 # C00[0]
    W[C_blocks[3][0], offset] += 1 # C11[0]
    # Note: C00 and C11 are 2x2 blocks.
    # M1 adds to C00 and C11.
    # C00[0] += 1, C00[1] += 0? 
    # Strassen formula:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 contributes 1 to all entries of C00? No.
    # M1 is a 2x2 matrix.
    # C00 = M1 + ... means C00[0] += M1[0], C00[1] += M1[1], etc.
    # So we need to add 1 to all 4 entries of C00 and C11.
    
    # Correction:
    for i in range(4):
        W[C_blocks[0][i], offset] += 1
        W[C_blocks[3][i], offset] += 1
    offset += 1
    
    # M2 = (A10+A11)B00
    for i in range(4):
        U[A_blocks[2][i], offset] += 1
        U[A_blocks[3][i], offset] += 1
        V[B_blocks[0][i], offset] += 1
    for i in range(4):
        W[C_blocks[2][i], offset] += 1 # C10
        W[C_blocks[3][i], offset] -= 1 # C11
    offset += 1
    
    # M3 = A00(B01-B11)
    for i in range(4):
        U[A_blocks[0][i], offset] += 1
        V[B_blocks[1][i], offset] += 1
        V[B_blocks[3][i], offset] -= 1
    for i in range(4):
        W[C_blocks[1][i], offset] += 1 # C01
        W[C_blocks[3][i], offset] += 1 # C11
    offset += 1
    
    # M4 = A11(B10-B00)
    for i in range(4):
        U[A_blocks[3][i], offset] += 1
        V[B_blocks[2][i], offset] += 1
        V[B_blocks[0][i], offset] -= 1
    for i in range(4):
        W[C_blocks[0][i], offset] += 1 # C00
        W[C_blocks[2][i], offset] += 1 # C10
    offset += 1
    
    # M5 = (A00+A01)(B11+B10)
    for i in range(4):
        U[A_blocks[0][i], offset] += 1
        U[A_blocks[1][i], offset] += 1
        V[B_blocks[3][i], offset] += 1
        V[B_blocks[2][i], offset] += 1
    for i in range(4):
        W[C_blocks[0][i], offset] -= 1 # C00
        W[C_blocks[1][i], offset] += 1 # C01
    offset += 1
    
    # M6 = (A10-A00)(B00+B01)
    for i in range(4):
        U[A_blocks[2][i], offset] += 1
        U[A_blocks[0][i], offset] -= 1
        V[B_blocks[0][i], offset] += 1
        V[B_blocks[1][i], offset] += 1
    for i in range(4):
        W[C_blocks[3][i], offset] += 1 # C11
    offset += 1
    
    # M7 = (A01-A11)(B10+B11)
    for i in range(4):
        U[A_blocks[1][i], offset] += 1
        U[A_blocks[3][i], offset] -= 1
        V[B_blocks[2][i], offset] += 1
        V[B_blocks[3][i], offset] += 1
    for i in range(4):
        W[C_blocks[0][i], offset] += 1 # C00
    offset += 1
    
    return offset
