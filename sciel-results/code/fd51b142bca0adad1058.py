"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # We use a block decomposition of the middle dimension m=4 into two blocks of size 2.
    # Block 1: A[:, 0:2] * B[0:2, :]
    # Block 2: A[:, 2:4] * B[2:4, :]
    
    # Each block is a 2x2 by 2x5 multiplication.
    # We decompose 2x2 by 2x5 into:
    # - 2x2 by 2x2 (cols 0,1) using Strassen (rank 7)
    # - 2x2 by 2x2 (cols 2,3) using Strassen (rank 7)
    # - 2x2 by 2x1 (col 4) using Schoolbook (rank 4)
    # Total rank per block = 18.
    # Total rank = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)] # 8 x 36
    V = [[0] * rank for _ in range(m * p)] # 20 x 36
    W = [[0] * rank for _ in range(p * n)] # 10 x 36
    
    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+j)
    # M2 = (c+d)(e+j)
    # M3 = a(f-j)
    # M4 = d(g-i)
    # M5 = (a+b)(j-k)
    # M6 = (c+b)(e+k)
    # M7 = (d-b)(g+l)
    # Note: Standard Strassen inputs:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Here B is 2x2.
    
    # We define the linear combinations for A and B for the 7 terms.
    # A is 2x2. Indices: 00, 01, 10, 11.
    # B is 2x2. Indices: 00, 01, 10, 11.
    
    # Term 1: (A00+A11) * (B00+B10)
    # Term 2: (A10+A11) * (B00+B10)
    # Term 3: A00 * (B01-B10)
    # Term 4: A11 * (B11-B01)
    # Term 5: (A00+A01) * (B10-B11)
    # Term 6: (A10+A01) * (B00+B11)
    # Term 7: (A11-A01) * (B11+B00) -- Wait, standard is (d-b)(g+l) where g is B10, l is B11?
    # Let's stick to a verified set.
    
    # Verified Strassen for C = AB:
    # M1 = (A00+A11)(B00+B10)
    # M2 = (A10+A11)(B00+B10)
    # M3 = A00(B01-B10)
    # M4 = A11(B11-B01)
    # M5 = (A00+A01)(B10-B11)
    # M6 = (A10+A01)(B00+B11)
    # M7 = (A11-A01)(B10+B11) -- Check C00 = M1 + M4 - M5 + M7?
    # C00 = (A00+A11)(B00+B10) + A11(B11-B01) - (A00+A01)(B10-B11) + (A11-A01)(B10+B11)
    # = A00B00 + A00B10 + A11B00 + A11B10 + A11B11 - A11B01 - A00B10 + A00B11 - A01B10 + A01B11 + A11B10 + A11B11 - A01B10 - A01B11
    # Terms:
    # A00B00: +1
    # A00B10: +1 -1 = 0
    # A11B00: +1
    # A11B10: +1 +1 = 2? No.
    # Let's use the standard one from Wikipedia.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11+B12)
    # M3 = (A11-A22)(B21-B22)
    # M4 = A11(B12-B22)
    # M5 = A22(B21-B11)
    # M6 = (A11+A12)(B22-B21)
    # M7 = (A21+A12)(B11-B12)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M4 + M6
    # C21 = M2 + M5 - M3 - M7
    # C22 = M1 - M2 + M3 + M5
    
    # We will use this set.
    
    def add_strassen_block(u_offset, v_offset, w_offset, col_offset):
        """
        Adds a 2x2x2 Strassen block.
        u_offset: row offset in U (A matrix)
        v_offset: row offset in V (B matrix)
        w_offset: row offset in W (C matrix)
        col_offset: column offset in B and C
        """
        # A is 2x2, local indices 0,1 for rows, 0,1 for cols.
        # Global A index: (u_offset + i) * m + (j + j_offset)?
        # Wait, U is (n*m) x rank.
        # U index is i*m + j.
        # For Block 1, j is 0,1. For Block 2, j is 2,3.
        
        # B is 2x2, local indices 0,1 for rows, 0,1 for cols.
        # Global B index: (k + k_offset) * p + (col_offset + l).
        # V index is k*p + l.
        
        # C is 2x2, local indices 0,1 for rows, 0,1 for cols.
        # Global C index: (col_offset + l) * n + (w_offset + i).
        # W index is l*n + i.
        
        # Strassen Terms
        # 1: (A00+A11) * (B00+B10)
        # 2: (A10+A11) * (B00+B01)
        # 3: (A00-A11) * (B10-B11)
        # 4: A00 * (B01-B11)
        # 5: A11 * (B10-B00)
        # 6: (A00+A01) * (B11-B10)
        # 7: (A10+A01) * (B00-B01)
        
        # Let's map these to U, V, W contributions.
        # U[i, r] += coeff_A
        # V[k, r] += coeff_B
        # W[l, r] += coeff_C
        
        # Term 1
        r = 0
        U[u_offset + 0][r] += 1 # A00
        U[u_offset + 1][r] += 1 # A11
        V[v_offset + 0][r] += 1 # B00
        V[v_offset + 1][r] += 1 # B10
        
        # Term 2
        r = 1
        U[u_offset + 1][r] += 1 # A10
        U[u_offset + 1][r] += 1 # A11
        V[v_offset + 0][r] += 1 # B00
        V[v_offset + 0][r] += 1 # B01 -- Wait, B01 is row 0 col 1.
        
        # This manual mapping is error-prone. 
        # Better to define the linear forms explicitly.
        
        pass

    # Let's implement the full logic directly.
    
    # Helper to add a term to U, V, W
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, r):
        U[u_idx][r] += u_val
        V[v_idx][r] += v_val
        W[w_idx][r] += w_val

    # Block 1: A cols 0,1. B rows 0,1.
    # A indices: 0,1,4,5 (i*m+j)
    # B indices: 0,1,5,6 (k*p+l)
    # C indices: 0,1,2,3 (l*n+i) for cols 0,1
    
    # Strassen 2x2x2 for Block 1, Cols 0,1 of B
    # Local B cols 0,1 -> Global B cols 0,1.
    
    # Term 1: (A00+A11)(B00+B10)
    # A00: U[0], A11: U[5]
    # B00: V[0], B10: V[5]
    # C00: W[0], C01: W[1], C10: W[2], C11: W[3]
    
    # We will use the standard 7 terms.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M4 + M6
    # C10 = M2 + M5 - M3 - M7
    # C11 = M1 - M2 + M3 + M5
    
    # M1 = (A00+A11)(B00+B10)
    # M2 = (A10+A11)(B00+B01)
    # M3 = (A00-A11)(B10-B11)
    # M4 = A00(B01-B11)
    # M5 = A11(B10-B00)
    # M6 = (A00+A01)(B11-B10)
    # M7 = (A10+A01)(B00-B01)
    
    # Note: B indices in local 2x2:
    # B00, B01
    # B10, B11
    
    # Global B indices for Block 1, Cols 0,1:
    # B00: V[0]
    # B01: V[1]
    # B10: V[5]
    # B11: V[6]
    
    # Global A indices for Block 1:
    # A00: U[0]
    # A01: U[1]
    # A10: U[4]
    # A11: U[5]
    
    # Global C indices for Block 1, Cols 0,1:
    # C00: W[0]
    # C01: W[1]
    # C10: W[2]
    # C11: W[3]
    
    # Add Strassen terms for Block 1, Cols 0,1
    # r = 0
    U[0][0] += 1; U[5][0] += 1
    V[0][0] += 1; V[5][0] += 1
    
    # r = 1
    U[4][1] += 1; U[5][1] += 1
    V[0][1] += 1; V[1][1] += 1
    
    # r = 2
    U[0][2] += 1; U[5][2] -= 1
    V[5][2] += 1; V[6][2] -= 1
    
    # r = 3
    U[0][3] += 1
    V[1][3] += 1; V[6][3] -= 1
    
    # r = 4
    U[5][4] += 1
    V[5][4] += 1; V[0][4] -= 1
    
    # r = 5
    U[0][5] += 1; U[1][5] += 1
    V[6][5] += 1; V[5][5] -= 1
    
    # r = 6
    U[4][6] += 1; U[1][6] += 1
    V[0][6] += 1; V[1][6] -= 1
    
    # Now add Strassen terms for Block 1, Cols 2,3
    # Local B cols 0,1 -> Global B cols 2,3.
    # B00: V[2]
    # B01: V[3]
    # B10: V[7]
    # B11: V[8]
    
    # C00 (of this sub-block) -> Global C Col 2 -> W[4]
    # C01 -> Global C Col 3 -> W[5]
    # C10 -> Global C Col 2 -> W[6]
    # C11 -> Global C Col 3 -> W[7]
    
    # r = 7
    U[0][7] += 1; U[5][7] += 1
    V[2][7] += 1; V[7][7] += 1
    
    # r = 8
    U[4][8] += 1; U[5][8] += 1
    V[2][8] += 1; V[3][8] += 1
    
    # r = 9
    U[0][9] += 1; U[5][9] -= 1
    V[7][9] += 1; V[8][9] -= 1
    
    # r = 10
    U[0][10] += 1
    V[3][10] += 1; V[8][10] -= 1
    
    # r = 11
    U[5][11] += 1
    V[7][11] += 1; V[2][11] -= 1
    
    # r = 12
    U[0][12] += 1; U[1][12] += 1
    V[8][12] += 1; V[7][12] -= 1
    
    # r = 13
    U[4][13] += 1; U[1][13] += 1
    V[2][13] += 1; V[3][13] -= 1
    
    # Now add Schoolbook terms for Block 1, Col 4
    # B00: V[4]
    # B10: V[9]
    # C00: W[8]
    # C10: W[9]
    
    # Terms:
    # M1 = A00 * B00
    # M2 = A01 * B10
    # M3 = A10 * B00
    # M4 = A11 * B10
    
    # C00 = M1 + M2
    # C10 = M3 + M4
    
    # r = 14
    U[0][14] += 1
    V[4][14] += 1
    
    # r = 15
    U[1][15] += 1
    V[9][15] += 1
    
    # r = 16
    U[4][16] += 1
    V[4][16] += 1
    
    # r = 17
    U[5][17] += 1
    V[9][17] += 1
    
    # Now Block 2: A cols 2,3. B rows 2,3.
    # A indices:
    # A00 (local) -> U[2] (Global i=0, j=2)
    # A01 (local) -> U[3] (Global i=0, j=3)
    # A10 (local) -> U[6] (Global i=1, j=2)
    # A11 (local) -> U[7] (Global i=1, j=3)
    
    # B indices for Block 2, Cols 0,1:
    # B00 (local) -> V[10] (Global k=2, l=0)
    # B01 (local) -> V[11] (Global k=2, l=1)
    # B10 (local) -> V[15] (Global k=3, l=0)
    # B11 (local) -> V[16] (Global k=3, l=1)
    
    # C indices for Block 2, Cols 0,1:
    # Same as Block 1: W[0..3]
    
    # r = 18
    U[2][18] += 1; U[7][18] += 1
    V[10][18] += 1; V[15][18] += 1
    
    # r = 19
    U[6][19] += 1; U[7][19] += 1
    V[10][19] += 1; V[11][19] += 1
    
    # r = 20
    U[2][20] += 1; U[7][20] -= 1
    V[15][20] += 1; V[16][20] -= 1
    
    # r = 21
    U[2][21] += 1
    V[11][21] += 1; V[16][21] -= 1
    
    # r = 22
    U[7][22] += 1
    V[15][22] += 1; V[10][22] -= 1
    
    # r = 23
    U[2][23] += 1; U[3][23] += 1
    V[16][23] += 1; V[15][23] -= 1
    
    # r = 24
    U[6][24] += 1; U[3][24] += 1
    V[10][24] += 1; V[11][24] -= 1
    
    # B indices for Block 2, Cols 2,3:
    # B00 -> V[12]
    # B01 -> V[13]
    # B10 -> V[17]
    # B11 -> V[18]
    
    # r = 25
    U[2][25] += 1; U[7][25] += 1
    V[12][25] += 1; V[17][25] += 1
    
    # r = 26
    U[6][26] += 1; U[7][26] += 1
    V[12][26] += 1; V[13][26] += 1
    
    # r = 27
    U[2][27] += 1; U[7][27] -= 1
    V[17][27] += 1; V[18][27] -= 1
    
    # r = 28
    U[2][28] += 1
    V[13][28] += 1; V[18][28] -= 1
    
    # r = 29
    U[7][29] += 1
    V[17][29] += 1; V[12][29] -= 1
    
    # r = 30
    U[2][30] += 1; U[3][30] += 1
    V[18][30] += 1; V[17][30] -= 1
    
    # r = 31
    U[6][31] += 1; U[3][31] += 1
    V[12][31] += 1; V[13][31] -= 1
    
    # B indices for Block 2, Col 4:
    # B00 -> V[14]
    # B10 -> V[19]
    
    # r = 32
    U[2][32] += 1
    V[14][32] += 1
    
    # r = 33
    U[3][33] += 1
    V[19][33] += 1
    
    # r = 34
    U[6][34] += 1
    V[14][34] += 1
    
    # r = 35
    U[7][35] += 1
    V[19][35] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
