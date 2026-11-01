<<<<<<< SEARCH
def try_rank(rank):
    """Try to find a rank-r decomposition using integer/half-integer search."""
    if rank < 9:
        return None
    
    # Construction: Strassen (rank 7) for columns 0,1 of B + 2 terms for column 2
    # Total rank = 9
    
    # A = [[a00, a01], [a10, a11]] indexed as A[0]=a00, A[1]=a01, A[2]=a10, A[3]=a11
    # B = [[b00, b01, b02], [b10, b11, b12]] indexed as B[0]=b00, B[1]=b01, B[2]=b02, B[3]=b10, B[4]=b11, B[5]=b12
    # C = AB, C = [[c00, c01, c02], [c10, c11, c12]]
    # W index: k*n + i, so W[0]=c00, W[1]=c01, W[2]=c02, W[3]=c10, W[4]=c11, W[5]=c12
    
    U = [[0] * rank for _ in range(4)]  # rows: a00, a01, a10, a11
    V = [[0] * rank for _ in range(6)]  # rows: b00, b01, b02, b10, b11, b12
    W = [[0] * rank for _ in range(6)]  # rows: c00, c01, c02, c10, c11, c12
    
    # Strassen's 7 terms for 2x2 x 2x2 (columns 0,1 of B):
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 0: M1
    U[0][0] = 1; U[3][0] = 1  # a00 + a11
    V[0][0] = 1; V[4][0] = 1  # b00 + b11
    W[0][0] = 1; W[3][0] = 1  # c00 + c11
    
    # Term 1: M2
    U[2][1] = 1; U[3][1] = 1  # a10 + a11
    V[0][1] = 1  # b00
    W[3][1] = 1; W[4][1] = -1  # c10 - c11
    
    # Term 2: M3
    U[0][2] = 1  # a00
    V[1][2] = 1; V[4][2] = -1  # b01 - b11
    W[1][2] = 1; W[4][2] = 1  # c01 + c11
    
    # Term 3: M4
    U[3][3] = 1  # a11
    V[3][3] = 1; V[0][3] = -1  # b10 - b00
    W[0][3] = 1; W[3][3] = 1  # c00 + c10
    
    # Term 4: M5
    U[0][4] = 1; U[1][4] = 1  # a00 + a01
    V[4][4] = 1  # b11
    W[0][4] = -1; W[1][4] = 1  # -c00 + c01
    
    # Term 5: M6
    U[2][5] = 1; U[0][5] = -1  # a10 - a00
    V[0][5] = 1; V[1][5] = 1  # b00 + b01
    W[4][5] = 1  # c11
    
    # Term 6: M7
    U[1][6] = 1; U[3][6] = -1  # a01 - a11
    V[3][6] = 1; V[4][6] = 1  # b10 + b11
    W[0][6] = 1  # c00
    
    # Terms 7,8: for column 2 of B (b02, b12)
    # c02 = a00*b02 + a01*b12
    # c12 = a10*b02 + a11*b12
    # We need 2 more terms to cover these 4 products with the existing A structure
    # Term 7: (a00+a10) * b02 -> contributes to c02 (via a00*b02) and c12 (via a10*b02)
    U[0][7] = 1; U[2][7] = 1  # a00 + a10
    V[2][7] = 1  # b02
    W[2][7] = 1; W[5][7] = 1  # c02 + c12
    
    # Term 8: (a01+a11) * b12 -> contributes to c02 (via a01*b12) and c12 (via a11*b12)
    U[1][8] = 1; U[3][8] = 1  # a01 + a11
    V[5][8] = 1  # b12
    W[2][8] = 1; W[5][8] = 1  # c02 + c12
    
    # Verify: c02 = a00*b02 + a01*b12
    # From term 7: a00*b02 + a10*b02
    # From term 8: a01*b12 + a11*b12
    # Sum: a00*b02 + a10*b02 + a01*b12 + a11*b12
    # But we need: a00*b02 + a01*b12 for c02, and a10*b02 + a11*b12 for c12
    # The cross terms a10*b02 in c02 and a01*b12 in c12 are WRONG!
    
    # I need to fix this. Let me use separate terms or a different approach.
    # Actually, for the third column, I need:
    # c02 = a00*b02 + a01*b12
    # c12 = a10*b02 + a11*b12
    # This is a 2x2 x 2x1 multiplication which has rank 4 (schoolbook) or possibly less.
    # Actually the tensor for 2x2 x 2x1 has rank 4 (it's just matrix-vector mult).
    # So I need 4 more terms, not 2. That would give rank 11.
    
    # Let me redo with rank 11:
    if rank == 9:
        # The above construction doesn't work for rank 9 because column 2 needs 4 terms
        # Let me try a different approach or return None
        return None
    
    if rank >= 11:
        # Use 4 more terms for column 2
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(6)]
        W = [[0] * rank for _ in range(6)]
        
        # Re-add Strassen terms 0-6
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[3][1] = 1; W[4][1] = -1
        
        U[0][2] = 1
        V[1][2] = 1; V[4][2] = -1
        W[1][2] = 1; W[4][2] = 1
        
        U[3][3] = 1
        V[3][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[3][3] = 1
        
        U[0][4] = 1; U[1][4] = 1
        V[4][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[4][5] = 1
        
        U[1][6] = 1; U[3][6] = -1
        V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1
        
        # Terms 7-10: schoolbook for column 2
        # c02 = a00*b02 + a01*b12
        # c12 = a10*b02 + a11*b12
        
        # Term 7: a00 * b02
        U[0][7] = 1
        V[2][7] = 1
        W[2][7] = 1  # c02
        
        # Term 8: a01 * b12
        U[1][8] = 1
        V[5][8] = 1
        W[2][8] = 1  # c02
        
        # Term 9: a10 * b02
        U[2][9] = 1
        V[2][9] = 1
        W[5][9] = 1  # c12
        
        # Term 10: a11 * b12
        U[3][10] = 1
        V[5][10] = 1
        W[5][10] = 1  # c12
        
        return {"U": U, "V": V, "W": W}
    
    return None
=======
def try_rank(rank):
    """Try to find a rank-r decomposition using integer/half-integer search."""
    if rank < 11:
        return None
    
    # Construction: Strassen (rank 7) for columns 0,1 of B + 4 terms for column 2
    # Total rank = 11
    
    # A = [[a00, a01], [a10, a11]] indexed as A[0]=a00, A[1]=a01, A[2]=a10, A[3]=a11
    # B = [[b00, b01, b02], [b10, b11, b12]] indexed as B[0]=b00, B[1]=b01, B[2]=b02, B[3]=b10, B[4]=b11, B[5]=b12
    # C = AB, C = [[c00, c01, c02], [c10, c11, c12]]
    # W index: k*n + i, so W[0]=c00, W[1]=c01, W[2]=c02, W[3]=c10, W[4]=c11, W[5]=c12
    
    U = [[0] * rank for _ in range(4)]  # rows: a00, a01, a10, a11
    V = [[0] * rank for _ in range(6)]  # rows: b00, b01, b02, b10, b11, b12
    W = [[0] * rank for _ in range(6)]  # rows: c00, c01, c02, c10, c11, c12
    
    # Strassen's 7 terms for 2x2 x 2x2 (columns 0,1 of B):
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 0: M1
    U[0][0] = 1; U[3][0] = 1  # a00 + a11
    V[0][0] = 1; V[4][0] = 1  # b00 + b11
    W[0][0] = 1; W[4][0] = 1  # c00 + c11
    
    # Term 1: M2
    U[2][1] = 1; U[3][1] = 1  # a10 + a11
    V[0][1] = 1  # b00
    W[3][1] = 1; W[4][1] = -1  # c10 - c11
    
    # Term 2: M3
    U[0][2] = 1  # a00
    V[1][2] = 1; V[4][2] = -1  # b01 - b11
    W[1][2] = 1; W[4][2] = 1  # c01 + c11
    
    # Term 3: M4
    U[3][3] = 1  # a11
    V[3][3] = 1; V[0][3] = -1  # b10 - b00
    W[0][3] = 1; W[3][3] = 1  # c00 + c10
    
    # Term 4: M5
    U[0][4] = 1; U[1][4] = 1  # a00 + a01
    V[4][4] = 1  # b11
    W[0][4] = -1; W[1][4] = 1  # -c00 + c01
    
    # Term 5: M6
    U[2][5] = 1; U[0][5] = -1  # a10 - a00
    V[0][5] = 1; V[1][5] = 1  # b00 + b01
    W[4][5] = 1  # c11
    
    # Term 6: M7
    U[1][6] = 1; U[3][6] = -1  # a01 - a11
    V[3][6] = 1; V[4][6] = 1  # b10 + b11
    W[0][6] = 1  # c00
    
    # Terms 7-10: schoolbook for column 2
    # c02 = a00*b02 + a01*b12
    # c12 = a10*b02 + a11*b12
    
    # Term 7: a00 * b02
    U[0][7] = 1
    V[2][7] = 1
    W[2][7] = 1  # c02
    
    # Term 8: a01 * b12
    U[1][8] = 1
    V[5][8] = 1
    W[2][8] = 1  # c02
    
    # Term 9: a10 * b02
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1  # c12
    
    # Term 10: a11 * b12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1  # c12
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE

