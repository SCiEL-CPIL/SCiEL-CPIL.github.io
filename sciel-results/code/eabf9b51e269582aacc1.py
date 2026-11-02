"""Matrix multiplication 3,4,4 - Strassen-like block decomposition."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 44
    
    col = 0
    # Top 2 rows (i=0,1) use Strassen: 4 blocks * 7 = 28 multiplications
    for bi in range(4):
        for s in range(7):
            _apply_strassen_block(U, V, W, col, bi, s)
            col += 1
    
    # Bottom row (i=2) uses schoolbook: 4 blocks * 4 = 16 multiplications
    for bj in range(4):
        for k in range(4):
            _apply_schoolbook_block(U, V, W, col, bj, k)
            col += 1
    
    return {"U": U, "V": V, "W": W}


def _apply_strassen_block(U, V, W, col, bi, s):
    """
    Apply a Strassen multiplication for block bi.
    Block partitioning:
    A (2x4) split into A_L (2x2) and A_R (2x2)
    B (4x4) split into B_TL (2x2), B_TR (2x2), B_BL (2x2), B_BR (2x2)
    Result C (2x4) = [C_L (2x2), C_R (2x2)]
    
    Blocks:
    bi=0: A_L * B_TL -> C_L  (rows 0-1, cols 0-1)
    bi=1: A_L * B_TR -> C_R  (rows 0-1, cols 2-3)
    bi=2: A_R * B_BL -> C_L  (rows 0-1, cols 0-1)
    bi=3: A_R * B_BR -> C_R  (rows 0-1, cols 2-3)
    """
    # Define the 4 sub-blocks of B for each bi
    # B_TL: rows 0-1, cols 0-1
    # B_TR: rows 0-1, cols 2-3
    # B_BL: rows 2-3, cols 0-1
    # B_BR: rows 2-3, cols 2-3
    
    block_defs = [
        # (A_subblock_rows, A_subblock_cols, B_subblock_rows, B_subblock_cols, C_rows, C_cols)
        (0, 0, 0, 0, 0, 0),  # bi=0: A_L * B_TL
        (0, 2, 0, 2, 0, 2),  # bi=1: A_R * B_TL... wait let me redo
    ]
    
    # Let me redefine more carefully.
    # A is 2x4: rows 0,1; cols 0,1,2,3
    #   A_L = A[0:2, 0:2]  (cols 0,1)
    #   A_R = A[0:2, 2:4]  (cols 2,3)
    # B is 4x4: rows 0,1,2,3; cols 0,1,2,3
    #   For bi=0: B block is B[0:2, 0:2] (TL), result C[0:2, 0:2]
    #   For bi=1: B block is B[0:2, 2:4] (TR), result C[0:2, 2:4]
    #   For bi=2: B block is B[2:4, 0:2] (BL), result C[0:2, 0:2]
    #   For bi=3: B block is B[2:4, 2:4] (BR), result C[0:2, 2:4]
    
    # For each bi, the A sub-block and B sub-block are 2x2
    if bi == 0:
        a_rows = (0, 1)  # A rows
        a_cols = (0, 1)  # A cols (left half)
        b_rows = (0, 1)  # B rows (top)
        b_cols = (0, 1)  # B cols (left)
        c_rows = (0, 1)  # C rows
        c_cols = (0, 1)  # C cols
    elif bi == 1:
        a_rows = (0, 1)
        a_cols = (0, 1)  # A cols (left half) - wait, for bi=1 we want A_L * B_TR
        b_rows = (0, 1)
        b_cols = (2, 3)  # B cols (right)
        c_rows = (0, 1)
        c_cols = (2, 3)
    elif bi == 2:
        a_rows = (0, 1)
        a_cols = (2, 3)  # A cols (right half) - A_R * B_BL
        b_rows = (2, 3)
        b_cols = (0, 1)
        c_rows = (0, 1)
        c_cols = (0, 1)
    else:  # bi == 3
        a_rows = (0, 1)
        a_cols = (2, 3)  # A cols (right half)
        b_rows = (2, 3)
        b_cols = (2, 3)
        c_rows = (0, 1)
        c_cols = (2, 3)
    
    # Strassen 2x2 multiplication:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    ar0, ar1 = a_rows
    ac0, ac1 = a_cols
    br0, br1 = b_rows
    bc0, bc1 = b_cols
    cr0, cr1 = c_rows
    cc0, cc1 = c_cols
    
    # Local indices: a[i][j] where i in {0,1} maps to global row ar0+i, col ac0+j
    #                b[i][j] where i in {0,1} maps to global row br0+i, col bc0+j
    #                c[i][j] where i in {0,1} maps to global row cr0+i, col cc0+j
    
    # We'll build the U, V, W contributions for this Strassen block.
    # Each of the 7 Strassen products is a bilinear form.
    
    # Helper: add to U, V, W for a term U_val * V_val contributing to W_val * C_position
    def add_term(u_row, u_val, v_row, v_val, w_row, w_val):
        U[u_row][col] += u_val
        V[v_row][col] += v_val
        W[w_row][col] += w_val
    
    # Global index helpers
    def u_idx(i, j):  # A[i][j], i in 0..1, j in 0..3
        return i * m + j
    
    def v_idx(i, j):  # B[i][j], i in 0..3, j in 0..3
        return i * p + j
    
    def w_idx(i, j):  # C[i][j], i in 0..3, j in 0..2 (W[k*n+i])
        # W index is k*n + i where k is col of C, i is row of C
        return j * n + i
    
    # For the local 2x2 blocks:
    # a(0,0) = A[ar0][ac0], a(0,1) = A[ar0][ac1]
    # a(1,0) = A[ar1][ac0], a(1,1) = A[ar1][ac1]
    # b(0,0) = B[br0][bc0], b(0,1) = B[br0][bc1]
    # b(1,0) = B[br1][bc0], b(1,1) = B[br1][bc1]
    # c(0,0) = C[cr0][cc0], c(0,1) = C[cr0][cc1]
    # c(1,0) = C[cr1][cc0], c(1,1) = C[cr1][cc1]
    
    # Define local-to-global mappings
    def a_gl(li, lj):
        return u_idx(ar0 + li, ac0 + lj)
    
    def b_gl(li, lj):
        return v_idx(br0 + li, bc0 + lj)
    
    def c_gl(li, lj):
        return w_idx(cr0 + li, cc0 + lj)
    
    # M1 = (a00+a11)(b00+b11)
    # Terms: a00*b00, a00*b11, a11*b00, a11*b11
    # Contribution to C: c00 += M1, c11 += M1
    # We need to add 4 separate bilinear terms, each contributing to both c00 and c11.
    # But each term in U,V,W is a single column, so we need one column per bilinear term.
    # Wait, no - each column represents one scalar multiplication.
    # For M1, we have 4 products, each contributing to c00 and c11.
    # But we can't combine them into one column because U and V entries would be sums.
    # Actually, we CAN: U entry for this column would be... no, that doesn't work
    # because U and V are fixed across all columns.
    #
    # Actually, the way tensor decomposition works:
    # C[i][j] = sum_k U[i*?][k] * V[?][k] * W[?][k]
    # More precisely: C = sum_k (U[:,k] outer V[:,k]) * W[:,k]
    # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    #
    # The matrix multiplication tensor is: T[A,B,C] = C = A*B
    # C[i][j] = sum_k A[i][k] * B[k][j]
    # In our notation: C[i][j] = sum_{k} A[i*m+k] * B[k*p+j]
    # So T = sum over (i,j,k) of e_{i*m+k} in U-space, e_{k*p+j} in V-space, e_{j*n+i} in W-space
    #
    # For a rank-1 term: U_val * V_val * W_val where
    #   U_val is at position corresponding to A element
    #   V_val is at position corresponding to B element  
    #   W_val is at position corresponding to C element
    #
    # For Strassen, each of the 7 M's is a sum of products, and each C is a sum of M's.
    # The key insight: we can express the entire 2x2 multiplication as 7 rank-1 terms
    # in the tensor decomposition, where each term has coefficients in U, V, W.
    
    # For M1 = (a00+a11)(b00+b11):
    # This is NOT a single bilinear term - it's a product of sums.
    # To decompose (x+y)(z+w) = xz + xw + yz + yw, we need 4 rank-1 terms.
    # But Strassen's trick is that we compute M1 as ONE multiplication of two linear combos.
    # In the tensor decomposition, this means:
    #   U column has 1s at positions a00 and a11 (coefficients 1 each)
    #   V column has 1s at positions b00 and b11 (coefficients 1 each)
    #   W column has coefficients that map M1 to c00 and c11
    #
    # So for M1: U[a00]=1, U[a11]=1, V[b00]=1, V[b11]=1
    #   W[c00] = 1, W[c11] = 1
    # This gives: U·V = (a00+a11)(b00+b11) = M1
    # And W[c00]*M1 contributes to C[0][0], W[c11]*M1 contributes to C[1][1]
    
    # Great, so each Strassen M corresponds to ONE column in the decomposition!
    
    # M1: (a00+a11)(b00+b11)
    U[a_gl(0,0)][col] += 1
    U[a_gl(1,1)][col] += 1
    V[b_gl(0,0)][col] += 1
    V[b_gl(1,1)][col] += 1
    W[c_gl(0,0)][col] += 1
    W[c_gl(1,1)][col] += 1
    
    # M2: (a10+a11)*b00
    U[a_gl(1,0)][col+1] += 1
    U[a_gl(1,1)][col+1] += 1
    V[b_gl(0,0)][col+1] += 1
    W[c_gl(1,0)][col+1] += 1
    W[c_gl(1,1)][col+1] += -1
    
    # M3: a00*(b01-b11)
    U[a_gl(0,0)][col+2] += 1
    V[b_gl(0,1)][col+2] += 1
    V[b_gl(1,1)][col+2] += -1
    W[c_gl(0,1)][col+2] += 1
    W[c_gl(1,1)][col+2] += 1
    
    # M4: a11*(b10-b00)
    U[a_gl(1,1)][col+3] += 1
    V[b_gl(1,0)][col+3] += 1
    V[b_gl(0,0)][col+3] += -1
    W[c_gl(0,0)][col+3] += 1
    W[c_gl(1,0)][col+3] += 1
    
    # M5: (a00+a01)*b11
    U[a_gl(0,0)][col+4] += 1
    U[a_gl(0,1)][col+4] += 1
    V[b_gl(1,1)][col+4] += 1
    W[c_gl(0,0)][col+4] += -1
    W[c_gl(0,1)][col+4] += 1
    
    # M6: (a10-a00)*(b00+b01)
    U[a_gl(1,0)][col+5] += 1
    U[a_gl(0,0)][col+5] += -1
    V[b_gl(0,0)][col+5] += 1
    V[b_gl(0,1)][col+5] += 1
    W[c_gl(1,1)][col+5] += 1
    
    # M7: (a01-a11)*(b10+b11)
    U[a_gl(0,1)][col+6] += 1
    U[a_gl(1,1)][col+6] += -1
    V[b_gl(1,0)][col+6] += 1
    V[b_gl(1,1)][col+6] += 1
    W[c_gl(0,0)][col+6] += 1
    W[c_gl(1,1)][col+6] += -1


def _apply_schoolbook_block(U, V, W, col, bj, k):
    """
    Apply schoolbook for bottom row (i=2).
    bj: which 2x2 block of B (0=TL, 1=TR, 2=BL, 3=BR)
    k: which column within that block (0 or 1)
    """
    n, m, p = 3, 4, 4
    
    if bj == 0:
        b_row = 0 + k
        b_col = 0
    elif bj == 1:
        b_row = 0 + k
        b_col = 2
    elif bj == 2:
        b_row = 2 + k
        b_col = 0
    else:  # bj == 3
        b_row = 2 + k
        b_col = 2
    
    # For bottom row i=2, we compute C[2][b_col] and C[2][b_col+1]
    # C[2][j] = A[2][b_row] * B[b_row][j] ... wait no
    # Schoolbook: C[i][j] = sum_k A[i][k] * B[k][j]
    # For i=2, j = b_col or b_col+1:
    # C[2][j] += A[2][b_row] * B[b_row][j]
    # But we're fixing one (b_row, j) pair per column.
    
    # Actually for schoolbook, each column corresponds to one (i, k, j) triple.
    # Here i=2 is fixed, k=b_row, j varies.
    # But we want to handle both j = b_col and j = b_col+1? No, one column per triple.
    
    # Let me reconsider: for the bottom row, we have 4 B columns (j=0,1,2,3)
    # and 4 B rows (k=0,1,2,3), giving 16 schoolbook terms.
    # Each term: A[2][k] * B[k][j] contributes to C[2][j]
    
    # So bj should index the (k, j) pair, not a block.
    # Let me redo: bj = 0..3 corresponds to k = 0,1,2,3
    # And k parameter is actually j = 0,1,2,3? No, that's 16 terms.
    
    # Wait, I defined bj as block (0=TL,1=TR,2=BL,3=BR) and k as column within block.
    # That gives 4*2=8, but I need 16. Let me fix this.
    
    # Actually, let me just iterate over all 16 (k,j) pairs for the bottom row.
    # I'll adjust the calling code.
    pass  # This function won't be used as-is; see construct()
# EVOLVE-BLOCK-END

# Let me rewrite construct() to be correct.
def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    # Top 2 rows: Strassen, 4 blocks * 7 = 28
    for bi in range(4):
        for s in range(7):
            _apply_strassen(U, V, W, col, bi)
            col += 1
    
    # Bottom row: schoolbook, 4 k-values * 4 j-values = 16
    for k in range(4):
        for j in range(4):
            _apply_schoolbook(U, V, W, col, k, j)
            col += 1
    
    return {"U": U, "V": V, "W": W}


def _apply_strassen(U, V, W, col, bi):
    n, m, p = 3, 4, 4
    
    if bi == 0:
        ar = (0, 1); ac = (0, 1); br = (0, 1); bc = (0, 1); cr = (0, 1); cc = (0, 1)
    elif bi == 1:
        ar = (0, 1); ac = (0, 1); br = (0, 1); bc = (2, 3); cr = (0, 1); cc = (2, 3)
    elif bi == 2:
        ar = (0, 1); ac = (2, 3); br = (2, 3); bc = (0, 1); cr = (0, 1); cc = (0, 1)
    else:
        ar = (0, 1); ac = (2, 3); br = (2, 3); bc = (2, 3); cr = (0, 1); cc = (2, 3)
    
    def a_gl(li, lj): return (ar[0]+li)*m + (ac[0]+lj)
    def b_gl(li, lj): return (br[0]+li)*p + (bc[0]+lj)
    def c_gl(li, lj): return (cc[0]+lj)*n + (cr[0]+li)
    
    # M1
    U[a_gl(0,0)][col] = 1; U[a_gl(1,1)][col] = 1
    V[b_gl(0,0)][col] = 1; V[b_gl(1,1)][col] = 1
    W[c_gl(0,0)][col] = 1; W[c_gl(1,1)][col] = 1
    
    # M2
    c2 = col + 1
    U[a_gl(1,0)][c2] = 1; U[a_gl(1,1)][c2] = 1
    V[b_gl(0,0)][c2] = 1
    W[c_gl(1,0)][c2] = 1; W[c_gl(1,1)][c2] = -1
    
    # M3
    c3 = col + 2
    U[a_gl(0,0)][c3] = 1
    V[b_gl(0,1)][c3] = 1; V[b_gl(1,1)][c3] = -1
    W[c_gl(0,1)][c3] = 1; W[c_gl(1,1)][c3] = 1
    
    # M4
    c4 = col + 3
    U[a_gl(1,1)][c4] = 1
    V[b_gl(1,0)][c4] = 1; V[b_gl(0,0)][c4] = -1
    W[c_gl(0,0)][c4] = 1; W[c_gl(1,0)][c4] = 1
    
    # M5
    c5 = col + 4
    U[a_gl(0,0)][c5] = 1; U[a_gl(0,1)][c5] = 1
    V[b_gl(1,1)][c5] = 1
    W[c_gl(0,0)][c5] = -1; W[c_gl(0,1)][c5] = 1
    
    # M6
    c6 = col + 5
    U[a_gl(1,0)][c6] = 1; U[a_gl(0,0)][c6] = -1
    V[b_gl(0,0)][c6] = 1; V[b_gl(0,1)][c6] = 1
    W[c_gl(1,1)][c6] = 1
    
    # M7
    c7 = col + 6
    U[a_gl(0,1)][c7] = 1; U[a_gl(1,1)][c7] = -1
    V[b_gl(1,0)][c7] = 1; V[b_gl(1,1)][c7] = 1
    W[c_gl(0,0)][c7] = 1; W[c_gl(1,1)][c7] = -1


def _apply_schoolbook(U, V, W, col, k, j):
    n, m, p = 3, 4, 4
    # C[2][j] += A[2][k] * B[k][j]
    U[2*m + k][col] = 1
    V[k*p + j][col] = 1
    W[j*n + 2][col] = 1
# EVOLVE-BLOCK-END
