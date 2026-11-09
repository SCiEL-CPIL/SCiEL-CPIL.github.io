"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block Strassen-like decomposition.
    # A is 2x4, split into A1 (2x2) and A2 (2x2)
    # B is 4x4, split into B11, B12, B21, B22 (all 2x2)
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    # We compute the 4 products using Strassen's algorithm (7 mults each).
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplications for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to the global indices.
    # A indices: 0..7 (2x4)
    # A1: rows 0,1; cols 0,1 -> indices 0,1,2,3
    # A2: rows 0,1; cols 2,3 -> indices 4,5,6,7
    # B indices: 0..15 (4x4)
    # B11: rows 0,1; cols 0,1 -> indices 0,1,4,5
    # B12: rows 0,1; cols 2,3 -> indices 2,3,6,7
    # B21: rows 2,3; cols 0,1 -> indices 8,9,12,13
    # B22: rows 2,3; cols 2,3 -> indices 10,11,14,15
    # C indices: 0..7 (2x4)
    # C1: rows 0,1; cols 0,1 -> indices 0,1,2,3 (W index k*n+i)
    # C2: rows 0,1; cols 2,3 -> indices 4,5,6,7
    
    # Helper to add a Strassen block
    def add_strassen(offset_u, offset_v, offset_w, sign_c1, sign_c2):
        # offset_u: base index in U for the A block (0 for A1, 4 for A2? No, U is 2x4)
        # Actually, U is indexed by i*m+j.
        # For A1 (cols 0,1), U indices are 0,1,2,3.
        # For A2 (cols 2,3), U indices are 4,5,6,7.
        
        # We define the 7 terms for a 2x2 multiplication.
        # Each term r in 0..6:
        # U_part: 2x2 matrix of coefficients for A
        # V_part: 2x2 matrix of coefficients for B
        # W_part: 2x2 matrix of coefficients for C
        
        # M1
        U1 = [[1, 0], [0, 1]] # A11, A22
        V1 = [[1, 1], [1, 1]] # B11, B22
        W1 = [[1, 1], [1, 1]] # C11, C12, C21, C22
        
        # M2
        U2 = [[0, 0], [1, 1]] # A21, A22
        V2 = [[1, 0], [0, 0]] # B11
        W2 = [[0, 0], [1, 0]] # C21
        
        # M3
        U3 = [[1, 0], [0, 0]] # A11
        V3 = [[0, 1], [0, -1]] # B12, -B22
        W3 = [[0, 1], [0, 0]] # C12
        
        # M4
        U4 = [[0, 0], [0, 1]] # A22
        V4 = [[0, -1], [1, 0]] # -B11, B21
        W4 = [[0, 0], [1, 0]] # C21
        
        # M5
        U5 = [[1, 1], [0, 0]] # A11, A12
        V5 = [[0, 0], [0, 1]] # B22
        W5 = [[0, 1], [0, 0]] # C12
        
        # M6
        U6 = [[-1, 0], [1, 0]] # -A11, A21
        V6 = [[1, 1], [0, 0]] # B11, B12
        W6 = [[0, 0], [1, 1]] # C21, C22
        
        # M7
        U7 = [[0, 1], [0, -1]] # A12, -A22
        V7 = [[0, 0], [1, 1]] # B21, B22
        W7 = [[0, 1], [1, 0]] # C12, C21
        
        terms = [
            (U1, V1, W1),
            (U2, V1, W2), # Wait, V1 is wrong for M2. V2 is B11.
            (U3, V3, W3),
            (U4, V4, W4),
            (U5, V5, W5),
            (U6, V6, W6),
            (U7, V7, W7)
        ]
        # Correcting V2: B11 is [[1,0],[0,0]] in 2x2 block.
        terms[1] = (U2, [[1,0],[0,0]], W2)
        
        # Map to global indices
        # A1 block: A indices 0,1,2,3. Local (r,c) -> global r*2+c? No, A is 2x4.
        # A1 is cols 0,1. A[r][c] -> U[r*4+c]? No, U is (n*m) x R.
        # Index is i*m+j. i is row 0,1. j is col 0,1.
        # So A1 local (r,c) -> global i=r, j=c. Index r*4+c? No, m=4. Index r*4+c.
        # A2 is cols 2,3. Local (r,c) -> global i=r, j=c+2. Index r*4+(c+2).
        
        # B11 block: B indices 0,1,4,5. Local (r,c) -> global row r, col c. Index r*4+c.
        # B12 block: B indices 2,3,6,7. Local (r,c) -> global row r, col c+2. Index r*4+(c+2).
        # B21 block: B indices 8,9,12,13. Local (r,c) -> global row r+2, col c. Index (r+2)*4+c.
        # B22 block: B indices 10,11,14,15. Local (r,c) -> global row r+2, col c+2. Index (r+2)*4+(c+2).
        
        # C1 block: C indices 0,1,2,3. Local (r,c) -> global row r, col c. W index k*n+i = c*2+r.
        # C2 block: C indices 4,5,6,7. Local (r,c) -> global row r, col c+2. W index (c+2)*2+r.
        
        for t in range(7):
            U_t, V_t, W_t = terms[t]
            col = t # relative column in this block of 7
            
            # Determine which B block
            # We need to know which B block we are multiplying with.
            # This function is called for specific pairs (A_block, B_block).
            pass

    # This generic approach is messy. Let's hardcode the 28 terms.
    # Terms 0-6: A1 * B11
    # Terms 7-13: A2 * B21
    # Terms 14-20: A1 * B12
    # Terms 21-27: A2 * B22
    
    def add_block(a_offset, b_offset, w_offset, start_col):
        # a_offset: 0 for A1 (cols 0,1), 2 for A2 (cols 2,3) in terms of j
        # b_offset: 0 for B11 (rows 0,1 cols 0,1), 2 for B12 (rows 0,1 cols 2,3)
        #           4 for B21 (rows 2,3 cols 0,1), 6 for B22 (rows 2,3 cols 2,3) in terms of j,k?
        # B index is j*p+k = j*4+k.
        # B11: j=0,1, k=0,1.
        # B12: j=0,1, k=2,3.
        # B21: j=2,3, k=0,1.
        # B22: j=2,3, k=2,3.
        
        # A index is i*m+j = i*4+j.
        # A1: i=0,1, j=0,1.
        # A2: i=0,1, j=2,3.
        
        # W index is k*n+i = k*2+i.
        # C1: k=0,1.
        # C2: k=2,3.
        
        # Strassen terms for local 2x2
        # Local A: i=0,1, j=0,1. Global A idx: i*4 + (j+a_offset)
        # Local B: j=0,1, k=0,1. Global B idx: (j+b_offset_row)*4 + (k+b_offset_col)
        # Note: b_offset for B11 is (0,0). B12 is (0,2). B21 is (2,0). B22 is (2,2).
        
        # We need to pass row_offset and col_offset for B.
        pass

    # Let's just do it manually for the 4 blocks.
    
    # Block 1: A1 * B11 -> C1
    # A1: i=0,1, j=0,1. U idx: i*4+j.
    # B11: j=0,1, k=0,1. V idx: j*4+k.
    # C1: k=0,1, i=0,1. W idx: k*2+i.
    
    # Strassen 7 terms for 2x2:
    # 1: (A00+A11)(B00+B11) -> C00 += 1, C01 += 1, C10 += 1, C11 += 1
    # 2: (A10+A11)B00 -> C10 += 1
    # 3: A00(B01-B11) -> C01 += 1
    # 4: A11(B10-B00) -> C10 += 1
    # 5: (A00+A01)B11 -> C01 += 1
    # 6: (A10-A00)(B00+B01) -> C10 += 1, C11 += 1
    # 7: (A01-A11)(B10+B11) -> C01 += 1, C10 += 1
    
    # Map to global:
    # A00 -> U[0], A01 -> U[1], A10 -> U[2], A11 -> U[3]
    # B00 -> V[0], B01 -> V[1], B10 -> V[4], B11 -> V[5]
    # C00 -> W[0], C01 -> W[1], C10 -> W[2], C11 -> W[3]
    
    # Term 1:
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[5][0] += 1
    W[0][0] += 1; W[1][0] += 1; W[2][0] += 1; W[3][0] += 1
    
    # Term 2:
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    
    # Term 3:
    U[0][2] += 1
    V[1][2] += 1; V[5][2] += -1
    W[1][2] += 1
    
    # Term 4:
    U[3][3] += 1
    V[4][3] += 1; V[0][3] += -1
    W[2][3] += 1
    
    # Term 5:
    U[0][4] += 1; U[1][4] += 1
    V[5][4] += 1
    W[1][4] += 1
    
    # Term 6:
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[2][5] += 1; W[3][5] += 1
    
    # Term 7:
    U[1][6] += 1; U[3][6] += -1
    V[4][6] += 1; V[5][6] += 1
    W[1][6] += 1; W[2][6] += 1

    # Block 2: A2 * B21 -> C1
    # A2: i=0,1, j=2,3. U idx: i*4+2, i*4+3.
    # B21: j=2,3, k=0,1. V idx: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13.
    # C1: k=0,1, i=0,1. W idx: 0,1,2,3.
    
    # A00->U[2], A01->U[3], A10->U[6], A11->U[7]
    # B00->V[8], B01->V[9], B10->V[12], B11->V[13]
    
    # Term 8:
    U[2][7] += 1; U[7][7] += 1
    V[8][7] += 1; V[13][7] += 1
    W[0][7] += 1; W[1][7] += 1; W[2][7] += 1; W[3][7] += 1
    
    # Term 9:
    U[6][8] += 1; U[7][8] += 1
    V[8][8] += 1
    W[2][8] += 1
    
    # Term 10:
    U[2][9] += 1
    V[9][9] += 1; V[13][9] += -1
    W[1][9] += 1
    
    # Term 11:
    U[7][10] += 1
    V[12][10] += 1; V[8][10] += -1
    W[2][10] += 1
    
    # Term 12:
    U[2][11] += 1; U[3][11] += 1
    V[13][11] += 1
    W[1][11] += 1
    
    # Term 13:
    U[6][12] += 1; U[2][12] += -1
    V[8][12] += 1; V[9][12] += 1
    W[2][12] += 1; W[3][12] += 1
    
    # Term 14:
    U[3][13] += 1; U[7][13] += -1
    V[12][13] += 1; V[13][13] += 1
    W[1][13] += 1; W[2][13] += 1

    # Block 3: A1 * B12 -> C2
    # A1: U[0..3]
    # B12: j=0,1, k=2,3. V idx: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7.
    # C2: k=2,3, i=0,1. W idx: 2*2+0=4, 2*2+1=5, 3*2+0=6, 3*2+1=7.
    
    # B00->V[2], B01->V[3], B10->V[6], B11->V[7]
    # C00->W[4], C01->W[5], C10->W[6], C11->W[7]
    
    # Term 15:
    U[0][14] += 1; U[3][14] += 1
    V[2][14] += 1; V[7][14] += 1
    W[4][14] += 1; W[5][14] += 1; W[6][14] += 1; W[7][14] += 1
    
    # Term 16:
    U[2][15] += 1; U[3][15] += 1
    V[2][15] += 1
    W[6][15] += 1
    
    # Term 17:
    U[0][16] += 1
    V[3][16] += 1; V[7][16] += -1
    W[5][16] += 1
    
    # Term 18:
    U[3][17] += 1
    V[6][17] += 1; V[2][17] += -1
    W[6][17] += 1
    
    # Term 19:
    U[0][18] += 1; U[1][18] += 1
    V[7][18] += 1
    W[5][18] += 1
    
    # Term 20:
    U[2][19] += 1; U[0][19] += -1
    V[2][19] += 1; V[3][19] += 1
    W[6][19] += 1; W[7][19] += 1
    
    # Term 21:
    U[1][20] += 1; U[3][20] += -1
    V[6][20] += 1; V[7][20] += 1
    W[5][20] += 1; W[6][20] += 1

    # Block 4: A2 * B22 -> C2
    # A2: U[2,3,6,7]
    # B22: j=2,3, k=2,3. V idx: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15.
    # C2: W[4,5,6,7]
    
    # B00->V[10], B01->V[11], B10->V[14], B11->V[15]
    
    # Term 22:
    U[2][21] += 1; U[7][21] += 1
    V[10][21] += 1; V[15][21] += 1
    W[4][21] += 1; W[5][21] += 1; W[6][21] += 1; W[7][21] += 1
    
    # Term 23:
    U[6][22] += 1; U[7][22] += 1
    V[10][22] += 1
    W[6][22] += 1
    
    # Term 24:
    U[2][23] += 1
    V[11][23] += 1; V[15][23] += -1
    W[5][23] += 1
    
    # Term 25:
    U[7][24] += 1
    V[14][24] += 1; V[10][24] += -1
    W[6][24] += 1
    
    # Term 26:
    U[2][25] += 1; U[3][25] += 1
    V[15][25] += 1
    W[5][25] += 1
    
    # Term 27:
    U[6][26] += 1; U[2][26] += -1
    V[10][26] += 1; V[11][26] += 1
    W[6][26] += 1; W[7][26] += 1
    
    # Term 28:
    U[3][27] += 1; U[7][27] += -1
    V[14][27] += 1; V[15][27] += 1
    W[5][27] += 1; W[6][27] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
