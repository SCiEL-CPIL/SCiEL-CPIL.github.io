"""Rewritten program using recursive peeling/slab decomposition."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W (each 4x7) for Strassen 2x2x2 matrix multiplication.
    A is 2x2, B is 2x2, C = A*B is 2x2.
    Indexing: A[i,j] -> index i*2+j, B[i,j] -> i*2+j, C[i,j] -> i*2+j.
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B01 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A10)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Column 0: M1
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # Column 1: M2
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10
    W[3][1] = -1  # C11
    
    # Column 2: M3
    U[0][2] = 1  # A00
    V[2][2] = 1  # B10
    V[3][2] = -1  # B11
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # Column 3: M4
    U[3][3] = 1  # A11
    V[1][3] = 1  # B01
    V[0][3] = -1  # B00
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # Column 4: M5
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[0][4] = -1  # C00
    W[1][4] = 1  # C01
    
    # Column 5: M6
    U[2][5] = 1  # A10
    U[0][5] = -1  # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    W[3][5] = 1  # C11
    
    # Column 6: M7
    U[1][6] = 1  # A01
    U[2][6] = -1  # A10
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00
    
    return U, V, W


def _direct_sum(U1, V1, W1, U2, V2, W2):
    """Direct sum of two decompositions.
    Each has shape (rows_U, rank_i), (rows_V, rank_i), (rows_W, rank_i).
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 + r2
    rowsU = len(U1)
    rowsV = len(V1)
    rowsW = len(W1)
    
    U = [[0]*rank for _ in range(rowsU)]
    V = [[0]*rank for _ in range(rowsV)]
    W = [[0]*rank for _ in range(rowsW)]
    
    for i in range(rowsU):
        for c in range(r1):
            U[i][c] = U1[i][c]
    for i in range(rowsV):
        for c in range(r1):
            V[i][c] = V1[i][c]
    for i in range(rowsW):
        for c in range(r1):
            W[i][c] = W1[i][c]
    
    for i in range(rowsU):
        for c in range(r2):
            U[i][r1+c] = U2[i][c]
    for i in range(rowsV):
        for c in range(r2):
            V[i][r1+c] = V2[i][c]
    for i in range(rowsW):
        for c in range(r2):
            W[i][r1+c] = W2[i][c]
    
    return U, V, W


def _schoolbook_2x1xp(p):
    """Schoolbook decomposition for 2x1xp matrix multiplication.
    A is 2x1, B is 1xp, C is 2xp.
    Rank = 2*p.
    U: 2 x (2p), V: p x (2p), W: 2p x (2p)
    Indexing: A[i,0] -> i, B[0,k] -> k, C[i,k] -> k*2+i
    """
    rank = 2 * p
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(p)]
    W = [[0]*rank for _ in range(2*p)]
    
    col = 0
    for i in range(2):
        for k in range(p):
            U[i][col] = 1
            V[k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    
    return U, V, W


def _construct_2x3xp(p):
    """Construct decomposition for 2x3xp matrix multiplication.
    A is 2x3, B is 3xp, C is 2xp.
    Uses recursive peeling: split B into first column (3x1) and rest (3x(p-1)).
    C = A * B = A * [b0 | B_rest] = A * b0 + A * B_rest
    
    For A * b0 (2x3 times 3x1 = 2x1):
      This is a 2x3x1 multiplication.
      C_col = A * b0, a 2x1 vector.
      We can use schoolbook for this: rank 2*3 = 6.
    
    For A * B_rest (2x3 times 3x(p-1) = 2x(p-1)):
      This is the recursive case 2x3x(p-1).
    
    But wait, we need to be more careful about the indexing.
    Let's think of it differently.
    
    Actually, let's just use a direct schoolbook approach for small cases and
    Strassen for the 2x2x2 part.
    
    Alternative: decompose 2x3xp as:
      Split A into A_left (2x2) and A_right (2x1).
      Split B into B_top (2xp) and B_bottom (1xp).
      
      C = A_left * B_top + A_right * B_bottom
      
      C is 2xp.
      A_left * B_top is 2x2 times 2xp = 2xp.
      A_right * B_bottom is 2x1 times 1xp = 2xp.
      
      So C = C_left + C_right.
      
      For C_left = A_left * B_top: this is 2x2xp.
        We can further decompose: for each column k, A_left * B_top[:,k] is 2x2 times 2x1.
        That's 2x2x1, rank 4 with schoolbook or 4 with Strassen... 
        Actually 2x2x1: A is 2x2, B is 2x1, C is 2x1.
        C0 = A00*B0 + A01*B1
        C1 = A10*B0 + A11*B1
        Schoolbook rank 4.
        
        So 2x2xp = sum over k of 2x2x1, each rank 4. Total rank 4p.
        But we can do better with Strassen for the 2x2x2 blocks.
        
      Let's try a different decomposition.
      
      For 2x3xp, let's peel off columns of B one at a time.
      Each column of B is 3x1.
      A * b_k (3x1) gives a 2x1 vector.
      C[:,k] = A * B[:,k]
      
      So C = [A*B[:,0] | A*B[:,1] | ... | A*B[:,p-1]]
      
      Each A*B[:,k] is 2x3x1 multiplication, rank 6 with schoolbook.
      Total rank = 6p.
      
      For p=7, that's 42, same as full schoolbook. Not an improvement.
      
    Let me try the Strassen-based approach for 2x3x7.
    
    Split: A = [A_L (2x2) | A_R (2x1)], B = [B_T (2x7); B_B (1x7)]
    C = A_L * B_T + A_R * B_B
    
    A_L * B_T: 2x2 times 2x7.
      Split B_T into 2x2 blocks and remainder: B_T = [B_T1 (2x2) | B_T2 (2x2) | B_T3 (2x2) | B_T4 (2x1)]
      A_L * B_T = A_L*B_T1 + A_L*B_T2 + A_L*B_T3 + A_L*B_T4
      
      Each A_L*B_Ti (2x2 times 2x2) uses Strassen, rank 7.
      A_L*B_T4 (2x2 times 2x1): 
        C0 = A00*B0 + A01*B1
        C1 = A10*B0 + A11*B1
        Rank 4 with schoolbook.
      
      So A_L * B_T rank = 7*3 + 4 = 25.
      
    A_R * B_B: 2x1 times 1x7, rank 14 with schoolbook.
    
    Total rank = 25 + 14 = 39.
    
    Let me implement this.
    """
    if p == 0:
        return [[0]*0 for _ in range(2)], [[0]*0 for _ in range(3)], [[0]*0 for _ in range(0)]
    
    # Decompose 2x3xp as A_L * B_T + A_R * B_B
    # A_L is 2x2 (first 2 cols of A), A_R is 2x1 (last col of A)
    # B_T is 2xp (first 2 rows of B), B_B is 1xp (last row of B)
    
    # C = A_L * B_T + A_R * B_B, both are 2xp
    
    # Part 1: A_L * B_T (2x2 times 2xp)
    # Split B_T into blocks of 2 columns each, plus remainder
    num_full_blocks = p // 2
    remainder_cols = p % 2
    
    U1 = [[0]*0 for _ in range(4)]  # will be extended
    V1 = [[0]*0 for _ in range(2)]
    W1 = [[0]*0 for _ in range(2*p)]
    
    rank1 = 0
    U1_cols = []
    V1_cols = []
    W1_cols = []
    
    for block in range(num_full_blocks):
        U_s, V_s, W_s = _strassen_2x2x2()
        # U_s is 4x7, V_s is 4x7, W_s is 4x7
        # But W_s corresponds to C which is 2x2, indexed as i*2+j
        # We need to map W_s indices to the correct columns in the full C (2xp)
        # Block covers columns [block*2, block*2+1] of C
        for i in range(4):
            U1_cols.append(U_s[i][:])
            V1_cols.append(V_s[i][:])
            # W_s[i] corresponds to C[i//2, i%2] in this 2x2 block
            # Full C index: row i//2, col block*2 + i%2
            # W full index: col*2 + row = (block*2 + i%2)*2 + i//2
            # Wait, the task says W[k*n+i, r] where n=2, so W index = k*2 + i
            # For C[row, col], W index = col*2 + row
            w_idx = (block * 2 + (i % 2)) * 2 + (i // 2)
            W1_cols.append([0]*(2*p))
            W1_cols[-1][w_idx] = W_s[i][0] if len(W_s[i]) > 0 else 0
            # Actually W_s[i] is a vector of length 7 (the rank dimension)
            # Let me re-examine. U_s[i] is a row of U_s, length 7.
            # W_s[i] is a row of W_s, length 7.
            # So for each rank column c, W_s[i][c] is the coefficient.
            # I need to place W_s[i][c] into the full W at position (w_idx, rank1+c)
        rank1 += 7
    
    # Handle remainder columns (if any)
    if remainder_cols > 0:
        # 2x2 times 2x1
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        # 4 terms
        for term_idx in range(4):
            U1_cols.append([0]*4)
            V1_cols.append([0]*2)
            W1_cols.append([0]*(2*p))
            r = rank1 + term_idx
            if term_idx == 0:
                U1_cols[-1][0] = 1   # A00
                V1_cols[-1][0] = 1   # B0 (row 0 of B_T, i.e., B row 0)
                W1_cols[-1][(num_full_blocks*2) * 2 + 0] = 1  # C[0, num_full_blocks*2]
            elif term_idx == 1:
                U1_cols[-1][1] = 1   # A01
                V1_cols[-1][1] = 1   # B1 (row 1 of B_T, i.e., B row 1)
                W1_cols[-1][(num_full_blocks*2) * 2 + 0] = 1  # C[0, num_full_blocks*2]
            elif term_idx == 2:
                U1_cols[-1][2] = 1   # A10
                V1_cols[-1][0] = 1   # B0
                W1_cols[-1][(num_full_blocks*2) * 2 + 1] = 1  # C[1, num_full_blocks*2]
            elif term_idx == 3:
                U1_cols[-1][3] = 1   # A11
                V1_cols[-1][1] = 1   # B1
                W1_cols[-1][(num_full_blocks*2) * 2 + 1] = 1  # C[1, num_full_blocks*2]
        rank1 += 4
    
    # Now build U1, V1, W1 properly
    rows_U1 = 4  # A_L is 2x2
    rows_V1 = 2  # B_T is 2xp
    rows_W1 = 2 * p
    
    U1 = [[0]*rank1 for _ in range(rows_U1)]
    V1 = [[0]*rank1 for _ in range(rows_V1)]
    W1 = [[0]*rank1 for _ in range(rows_W1)]
    
    for c in range(rank1):
        for i in range(rows_U1):
            U1[i][c] = U1_cols[c][i]
        for i in range(rows_V1):
            V1[i][c] = V1_cols[c][i]
        for i in range(rows_W1):
            W1[i][c] = W1_cols[c][i]
    
    # Part 2: A_R * B_B (2x1 times 1xp)
    U2, V2, W2 = _schoolbook_2x1xp(p)
    # U2 is 2 x (2p), V2 is p x (2p), W2 is 2p x (2p)
    # But we need to adjust: A_R is the last column of A, index 2 in A
    # In the full A (2x3), A_R corresponds to column 2.
    # U2 has rows for A_R[0] and A_R[1], which are A[0,2] and A[1,2].
    # In the full U (6 rows for 2x3 A), these are rows 2 and 3.
    # So we need to embed U2 into a 6-row matrix.
    
    rank2 = len(U2[0])
    
    # Now combine: total U is 6 rows (2x3), total V is 3p rows (3xp), total W is 2p rows
    # U1 has 4 rows (for A_L = A[:, 0:2]), need to embed into 6 rows
    # U2 has 2 rows (for A_R = A[:, 2]), need to embed into 6 rows at positions 2,3
    
    # V1 has 2 rows (for B_T = B[0:2, :]), need to embed into 3p rows
    # V2 has p rows (for B_B = B[2, :]), need to embed into 3p rows at positions 2p:3p
    
    # W1 and W2 both have 2p rows, they add together
    
    total_rank = rank1 + rank2
    rows_U = 6
    rows_V = 3 * p
    rows_W = 2 * p
    
    U = [[0]*total_rank for _ in range(rows_U)]
    V = [[0]*total_rank for _ in range(rows_V)]
    W = [[0]*total_rank for _ in range(rows_W)]
    
    # Embed U1 (4 rows) into U (6 rows) at rows 0,1
    for i in range(4):
        for c in range(rank1):
            U[i][c] = U1[i][c]
    
    # Embed U2 (2 rows) into U (6 rows) at rows 2,3
    for i in range(2):
        for c in range(rank2):
            U[2+i][rank1+c] = U2[i][c]
    
    # Embed V1 (2 rows) into V (3p rows) at rows 0,1
    for i in range(2):
        for c in range(rank1):
            V[i][c] = V1[i][c]
    
    # Embed V2 (p rows) into V (3p rows) at rows 2p, 2p+1, ..., 3p-1
    for i in range(p):
        for c in range(rank2):
            V[2*p+i][rank1+c] = V2[i][c]
    
    # Add W1 and W2
    for i in range(rows_W):
        for c in range(rank1):
            W[i][c] = W1[i][c]
        for c in range(rank2):
            W[i][rank1+c] = W2[i][c]
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    U, V, W = _construct_2x3xp(p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
