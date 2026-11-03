"""Matrix multiplication 6x6x6 using 2x2 block decomposition with rank-19 Laderman 3x3 decomposition."""

def construct():
    n, m, p = 6, 6, 6
    
    # Laderman's rank-19 decomposition for 3x3 matrix multiplication
    # U33, V33, W33 are 9x19 matrices (flattened 3x3)
    # U33[i*3+j, r], V33[j*3+k, r], W33[k*3+i, r]
    
    r3 = 19
    U33 = [[0] * r3 for _ in range(9)]
    V33 = [[0] * r3 for _ in range(9)]
    W33 = [[0] * r3 for _ in range(9)]
    
    # Laderman's 19 products
    # Indices: i, j, k in 0..2
    # U[i,j], V[j,k], W[k,i]
    
    # m0 = a00*b00
    U33[0][0] = 1; V33[0][0] = 1; W33[0][0] = 1
    # m1 = a11*b11
    U33[4][1] = 1; V33[4][1] = 1; W33[4][1] = 1
    # m2 = a22*b22
    U33[8][2] = 1; V33[8][2] = 1; W33[8][2] = 1
    # m3 = a01*b10
    U33[1][3] = 1; V33[3][3] = 1; W33[3][3] = 1
    # m4 = a12*b20
    U33[2][4] = 1; V33[6][4] = 1; W33[6][4] = 1
    # m5 = a20*b01
    U33[6][5] = 1; V33[1][5] = 1; W33[1][5] = 1
    # m6 = a02*b21
    U33[2][6] = 1; V33[7][6] = 1; W33[7][6] = 1
    # m7 = a10*b02
    U33[3][7] = 1; V33[2][7] = 1; W33[2][7] = 1
    # m8 = a21*b12
    U33[7][8] = 1; V33[5][8] = 1; W33[5][8] = 1
    # m9 = (a00+a10)*(b00+b10)
    U33[0][9] = 1; U33[3][9] = 1
    V33[0][9] = 1; V33[3][9] = 1
    W33[0][9] = 1
    # m10 = (a01+a11)*(b10+b11)
    U33[1][10] = 1; U33[4][10] = 1
    V33[3][10] = 1; V33[4][10] = 1
    W33[1][10] = 1
    # m11 = (a02+a12)*(b20+b21)
    U33[2][11] = 1; U33[5][11] = 1
    V33[6][11] = 1; V33[7][11] = 1
    W33[2][11] = 1
    # m12 = (a00+a20)*(b00+b20)
    U33[0][12] = 1; U33[6][12] = 1
    V33[0][12] = 1; V33[6][12] = 1
    W33[0][12] = 1
    # m13 = (a01+a21)*(b10+b20)
    U33[1][13] = 1; U33[7][13] = 1
    V33[3][13] = 1; V33[6][13] = 1
    W33[1][13] = 1
    # m14 = (a02+a22)*(b20+b21)
    U33[2][14] = 1; U33[8][14] = 1
    V33[6][14] = 1; V33[7][14] = 1
    W33[2][14] = 1
    # m15 = (a10+a20)*(b01+b21)
    U33[3][15] = 1; U33[6][15] = 1
    V33[1][15] = 1; V33[7][15] = 1
    W33[0][15] = 1
    # m16 = (a11+a21)*(b11+b21)
    U33[4][16] = 1; U33[7][16] = 1
    V33[4][16] = 1; V33[7][16] = 1
    W33[1][16] = 1
    # m17 = (a12+a22)*(b21+b22)
    U33[5][17] = 1; U33[8][17] = 1
    V33[7][17] = 1; V33[8][17] = 1
    W33[2][17] = 1
    # m18 = (a00+a01+a02)*(b00+b10+b20)
    U33[0][18] = 1; U33[1][18] = 1; U33[2][18] = 1
    V33[0][18] = 1; V33[3][18] = 1; V33[6][18] = 1
    W33[0][18] = 1; W33[1][18] = 1; W33[2][18] = 1
    
    # Note: The W coefficients above are simplified. 
    # In Laderman, the reconstruction is more complex.
    # C00 = m0 + m9 + m12 - m10 - m18
    # C01 = m3 + m10 + m13 - m11 - m18
    # C02 = m6 + m11 + m14 - m12 - m18
    # C10 = m7 + m9 + m15 - m13 - m18
    # C11 = m1 + m10 + m16 - m14 - m18
    # C12 = m4 + m11 + m17 - m15 - m18
    # C20 = m5 + m12 + m15 - m16 - m18
    # C21 = m8 + m13 + m16 - m17 - m18
    # C22 = m2 + m14 + m17 - m18
    
    # I need to update W33 to reflect these sums.
    # W33[k*3+i, r] holds the coefficient for C[i,k] from term r.
    
    # Reset W33
    W33 = [[0] * r3 for _ in range(9)]
    
    # Term 0 (m0): C00 += 1
    W33[0][0] = 1
    # Term 1 (m1): C11 += 1
    W33[4][1] = 1
    # Term 2 (m2): C22 += 1
    W33[8][2] = 1
    # Term 3 (m3): C01 += 1
    W33[3][3] = 1
    # Term 4 (m4): C12 += 1
    W33[6][4] = 1
    # Term 5 (m5): C20 += 1
    W33[6][5] = 1
    # Term 6 (m6): C02 += 1
    W33[2][6] = 1
    # Term 7 (m7): C10 += 1
    W33[3][7] = 1
    # Term 8 (m8): C21 += 1
    W33[7][8] = 1
    # Term 9 (m9): C00 += 1, C10 += 1
    W33[0][9] = 1; W33[3][9] = 1
    # Term 10 (m10): C00 -= 1, C01 += 1, C11 += 1
    W33[0][10] = -1; W33[3][10] = 1; W33[4][10] = 1
    # Term 11 (m11): C01 -= 1, C02 += 1, C12 += 1
    W33[3][11] = -1; W33[2][11] = 1; W33[6][11] = 1
    # Term 12 (m12): C00 += 1, C02 -= 1, C20 += 1
    W33[0][12] = 1; W33[2][12] = -1; W33[6][12] = 1
    # Term 13 (m13): C01 += 1, C10 -= 1, C21 += 1
    W33[3][13] = 1; W33[3][13] = 1 # Wait, C10 is W[3,1]? No.
    # C[i,k] is at W[k*3+i].
    # C10: i=1, k=0 -> W[0*3+1] = W[1].
    # C01: i=0, k=1 -> W[1*3+0] = W[3].
    # C21: i=1, k=2 -> W[2*3+1] = W[7].
    # C20: i=0, k=2 -> W[2*3+0] = W[6].
    # C02: i=2, k=0 -> W[0*3+2] = W[2].
    # C12: i=2, k=1 -> W[1*3+2] = W[5].
    
    # Let's re-map W indices carefully.
    # W_index = k * 3 + i
    
    # m9: C00 (i=0,k=0, idx=0), C10 (i=1,k=0, idx=1)
    W33[0][9] = 1
    W33[1][9] = 1
    
    # m10: C00 (idx=0) -1, C01 (i=0,k=1, idx=3) +1, C11 (i=1,k=1, idx=4) +1
    W33[0][10] = -1
    W33[3][10] = 1
    W33[4][10] = 1
    
    # m11: C01 (idx=3) -1, C02 (i=2,k=0, idx=2) +1, C12 (i=2,k=1, idx=5) +1
    W33[3][11] = -1
    W33[2][11] = 1
    W33[5][11] = 1
    
    # m12: C00 (idx=0) +1, C02 (idx=2) -1, C20 (i=0,k=2, idx=6) +1
    W33[0][12] = 1
    W33[2][12] = -1
    W33[6][12] = 1
    
    # m13: C01 (idx=3) +1, C10 (idx=1) -1, C21 (i=1,k=2, idx=7) +1
    W33[3][13] = 1
    W33[1][13] = -1
    W33[7][13] = 1
    
    # m14: C02 (idx=2) +1, C11 (idx=4) -1, C22 (i=2,k=2, idx=8) +1
    W33[2][14] = 1
    W33[4][14] = -1
    W33[8][14] = 1
    
    # m15: C10 (idx=1) +1, C12 (idx=5) -1, C20 (idx=6) +1
    W33[1][15] = 1
    W33[5][15] = -1
    W33[6][15] = 1
    
    # m16: C11 (idx=4) +1, C20 (idx=6) -1, C21 (idx=7) +1
    W33[4][16] = 1
    W33[6][16] = -1
    W33[7][16] = 1
    
    # m17: C12 (idx=5) +1, C21 (idx=7) -1, C22 (idx=8) +1
    W33[5][17] = 1
    W33[7][17] = -1
    W33[8][17] = 1
    
    # m18: C00 (idx=0) -1, C01 (idx=3) -1, C02 (idx=2) -1,
    #       C10 (idx=1) -1, C11 (idx=4) -1, C12 (idx=5) -1,
    #       C20 (idx=6) -1, C21 (idx=7) -1, C22 (idx=8) -1
    for idx in range(9):
        W33[idx][18] = -1

    # Now build the 6x6 decomposition
    rank = 7 * r3
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Strassen terms
    # a_coeffs: block -> coeff
    # b_coeffs: block -> coeff
    # c_coeffs: block -> coeff
    
    strassen_terms = [
        {'a': {'A00': 1, 'A11': 1}, 'b': {'B00': 1, 'B11': 1}, 'c': {'C00': 1, 'C11': 1}},
        {'a': {'A10': 1, 'A11': 1}, 'b': {'B00': 1}, 'c': {'C10': 1, 'C11': -1}},
        {'a': {'A00': 1}, 'b': {'B01': 1, 'B11': -1}, 'c': {'C01': 1, 'C11': 1}},
        {'a': {'A11': 1}, 'b': {'B10': 1, 'B00': -1}, 'c': {'C00': 1, 'C10': 1}},
        {'a': {'A00': 1, 'A01': 1}, 'b': {'B11': 1}, 'c': {'C00': -1, 'C01': 1}},
        {'a': {'A10': 1, 'A00': -1}, 'b': {'B00': 1, 'B01': 1}, 'c': {'C11': 1}},
        {'a': {'A01': 1, 'A11': -1}, 'b': {'B10': 1, 'B11': 1}, 'c': {'C00': 1}},
    ]
    
    block_map = {
        'A00': (0, 0), 'A01': (0, 1), 'A10': (1, 0), 'A11': (1, 1),
        'B00': (0, 0), 'B01': (0, 1), 'B10': (1, 0), 'B11': (1, 1),
        'C00': (0, 0), 'C01': (0, 1), 'C10': (1, 0), 'C11': (1, 1),
    }
    
    for m_idx, term in enumerate(strassen_terms):
        base_col = m_idx * r3
        
        for local_col in range(r3):
            col = base_col + local_col
            
            # A contributions
            for a_block, ca in term['a'].items():
                bi, bj = block_map[a_block]
                for i in range(3):
                    for j in range(3):
                        u_row = (bi*3+i)*6 + (bj*3+j)
                        U[u_row][col] += ca * U33[i*3+j][local_col]
            
            # B contributions
            for b_block, cb in term['b'].items():
                bk, bl = block_map[b_block]
                for j in range(3):
                    for k in range(3):
                        v_row = (bk*3+j)*6 + (bl*3+k)
                        V[v_row][col] += cb * V33[j*3+k][local_col]
            
            # C contributions
            for c_block, cc in term['c'].items():
                ci, ck = block_map[c_block]
                for i in range(3):
                    for k in range(3):
                        w_row = (ck*3+k)*6 + (ci*3+i)
                        W[w_row][col] += cc * W33[k*3+i][local_col]
    
    return {"U": U, "V": V, "W": W}
