"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _schoolbook_nm_p(n, m, p, offset):
    """Return list of (u_row, v_row, w_row) triples for schoolbook n x m x p."""
    triples = []
    col = offset
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                triples.append((u_row, v_row, w_row, col))
                col += 1
    return triples

def _strassen_2x2x2_block(a_block_row_offset, b_block_row_offset, c_block_row_offset, n, p, col_start):
    """
    Strassen for a 2x2x2 sub-problem.
    A is 2x2, B is 2x2, C is 2x2.
    a_block_row_offset: row offset in A for this 2x2 block (0 or 2 for the 2 rows of A)
    b_block_row_offset: row offset in B for this 2x2 block (0 or 2 for the 2 rows of B)
    c_block_row_offset: (c_row, c_col) offset for the 2x2 C block
    Returns list of triples and number of columns used.
    """
    # A rows: a_block_row_offset, a_block_row_offset+1
    # A cols: 0,1 (local) -> global col = 0,1
    # B rows: b_block_row_offset, b_block_row_offset+1
    # B cols: 0,1 (local) -> global col = 0,1
    # C rows: c_block_row_offset[0], c_block_row_offset[0]+1
    # C cols: c_block_row_offset[1], c_block_row_offset[1]+1
    
    a00 = a_block_row_offset * 2 + 0  # row 0 of local 2x2 A -> global A row
    a01 = a_block_row_offset * 2 + 1  # row 1 of local 2x2 A -> global A row
    a10 = (a_block_row_offset + 1) * 2 + 0  # wait, A is 2x2, so rows are 0,1
    # Actually, A is 2x2. The rows are indexed 0,1.
    # The columns of A are 0,1 (local to the 2x2 block).
    # But in the global A (n x m), if we're taking a 2x2 block, the rows are a_block_row_offset and a_block_row_offset+1,
    # and the columns are 0 and 1 (since we're splitting along the m dimension for A? No, A is n x m).
    # Let's re-examine the splitting.
    
    # For 3x4x7:
    # Split A (3x4) into A_top (2x4) and A_bot (1x4).
    # For A_top (2x4), split into A_11 (2x2) and A_12 (2x2) along columns.
    # So A_11 has rows 0,1 and cols 0,1.
    # A_12 has rows 0,1 and cols 2,3.
    
    # Split B (4x7) into B_11 (2x7) and B_12 (2x7) along rows.
    # B_11 has rows 0,1 and cols 0..6.
    # B_12 has rows 2,3 and cols 0..6.
    
    # A_11 * B_11: A is 2x2, B is 2x7.
    # We want to compute C_11 (2x7) = A_11 * B_11.
    # We use Strassen on the first 2 columns of B_11, and schoolbook on the remaining 5.
    # Strassen on A_11 (2x2) and B_11[:, 0:2] (2x2).
    
    # So for the Strassen part:
    # A rows: 0, 1 (global rows 0,1 since A_11 is top-left)
    # A cols: 0, 1 (global cols 0,1)
    # B rows: 0, 1 (global rows 0,1 since B_11 is top part)
    # B cols: 0, 1 (global cols 0,1)
    # C rows: 0, 1 (global rows 0,1)
    # C cols: 0, 1 (global cols 0,1)
    
    # Let me define the Strassen triples more carefully.
    # M1 = A00 * (B00 + B11)
    # M2 = (A00 + A11) * B10
    # M3 = (A10 + A11) * B00
    # M4 = A11 * (B11 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Where A00, A01, A10, A11 are the 1x1 entries of A (but A is 2x2, so these are scalars in the tensor sense)
    # In the tensor decomposition, each "multiplication" corresponds to a rank-1 term.
    # For A (2x2), the entries are:
    # A00 at (row=0, col=0) -> u_row = 0*2+0 = 0, v_row = 0*2+0 = 0
    # A01 at (row=0, col=1) -> u_row = 0*2+1 = 1, v_row = 0*2+1 = 1
    # A10 at (row=1, col=0) -> u_row = 1*2+0 = 2, v_row = 1*2+0 = 2
    # A11 at (row=1, col=1) -> u_row = 1*2+1 = 3, v_row = 1*2+1 = 3
    
    # But wait, in the global problem, A is n x m. Here we're considering a sub-block.
    # The U matrix has dimensions (n*m) x rank. Each row of U corresponds to an entry of A.
    # For the sub-block A_11 (2x2), the entries are at global positions:
    # (i=0, j=0), (i=0, j=1), (i=1, j=0), (i=1, j=1)
    # So u_rows are: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5 (if m=4)
    
    # Let me use a more general approach. I'll define the Strassen decomposition for a 2x2x2 block
    # where the local A is at rows a_rows, cols a_cols, local B is at rows b_rows, cols b_cols,
    # and local C is at rows c_rows, cols c_cols.
    
    # Actually, let me just hardcode the three Strassen blocks since the dimensions are fixed.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Strategy: 
    # 1. A_bot (1x4) * B (4x7): schoolbook, rank 4*7 = 28
    # 2. A_top (2x4) * B (4x7): split into A_11(2x2)*B_11(2x7) and A_12(2x2)*B_12(2x7)
    #    For each 2x2x7 block, use Strassen on first 2 cols (rank 7) + schoolbook on last 5 cols (rank 20) = 27
    # Total: 28 + 27 + 27 = 82
    
    # Let's build the decomposition.
    # U has shape (n*m, rank) = (12, 82)
    # V has shape (m*p, rank) = (28, 82)
    # W has shape (p*n, rank) = (21, 82)
    
    rank = 82
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_bot (row 2 of A) * B (all 4x7)
    # A_bot is row 2 of A, so i=2, j=0..3
    # For each j in 0..3, k in 0..6:
    #   U[2*m + j][col] = 1
    #   V[j*p + k][col] = 1
    #   W[k*n + 2][col] = 1
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    # col is now 28
    
    # Part 2: A_11 (rows 0,1; cols 0,1 of A) * B_11 (rows 0,1 of B; all 7 cols)
    # A_11 entries: (0,0), (0,1), (1,0), (1,1) -> u_rows: 0, 1, 4, 5
    # B_11 entries: (0,k), (1,k) for k=0..6 -> v_rows: k, 14+k for k=0..6
    # C_11 entries: (0,k), (1,k) for k=0..6 -> w_rows: k*3+0, k*3+1 for k=0..6
    
    # Strassen on first 2 cols of B_11 (k=0,1)
    # A_11 is 2x2, B_11[:,0:2] is 2x2
    # Local A: rows 0,1; cols 0,1 -> global u_rows: 0, 1, 4, 5
    # Local B: rows 0,1; cols 0,1 -> global v_rows: 0, 14, 1, 15
    # Local C: rows 0,1; cols 0,1 -> global w_rows: 0, 3, 1, 4
    
    # M1 = A00 * (B00 + B11)
    # A00: u_row=0, v_row=0
    # B00: v_row=0
    # B11: v_row=14 (row 1 of B_11, col 1 of local B -> global v_row = 1*7+1 = 8? Wait)
    # B is 4x7. B_11 is rows 0,1 of B. So B_11[i][k] for i=0,1, k=0..6.
    # v_row for B[i][k] is i*p + k = i*7 + k.
    # B00: i=0, k=0 -> v_row=0
    # B11: i=1, k=1 -> v_row=7+1=8
    # B10: i=1, k=0 -> v_row=7+0=7
    # B01: i=0, k=1 -> v_row=0+1=1
    
    # A00: i=0, j=0 -> u_row=0*4+0=0
    # A01: i=0, j=1 -> u_row=0*4+1=1
    # A10: i=1, j=0 -> u_row=1*4+0=4
    # A11: i=1, j=1 -> u_row=1*4+1=5
    
    # C00: i=0, k=0 -> w_row=0*3+0=0
    # C01: i=0, k=1 -> w_row=1*3+0=3
    # C10: i=1, k=0 -> w_row=0*3+1=1
    # C11: i=1, k=1 -> w_row=1*3+1=4
    
    # M1 = A00 * (B00 + B11): 
    #   u_row=0, v_row=0 and v_row=8, w_row=0
    # This requires two rank-1 terms:
    # Term 1: U[0][col]=1, V[0][col]=1, W[0][col]=1
    # Term 2: U[0][col]=1, V[8][col]=1, W[0][col]=1
    
    def add_term(u_row, v_row, w_row, val=1):
        nonlocal col
        U[u_row][col] += val
        V[v_row][col] += val
        W[w_row][col] += val
        col += 1
    
    # M1: A00 * B00 + A00 * B11
    add_term(0, 0, 0)  # A00*B00 -> C00
    add_term(0, 8, 0)  # A00*B11 -> C00
    
    # M2: (A00 + A11) * B10
    add_term(0, 7, 1)  # A00*B10 -> C10
    add_term(5, 7, 1)  # A11*B10 -> C10
    
    # M3: (A10 + A11) * B00
    add_term(4, 0, 0)  # A10*B00 -> C00
    add_term(5, 0, 0)  # A11*B00 -> C00
    
    # M4: A11 * (B11 - B00)
    add_term(5, 8, 4)  # A11*B11 -> C11
    add_term(5, 0, 4, -1)  # -A11*B00 -> C11
    
    # M5: (A00 + A01) * B11
    add_term(0, 8, 3)  # A00*B11 -> C01
    add_term(1, 8, 3)  # A01*B11 -> C01
    
    # M6: (A10 - A00) * (B00 + B01)
    add_term(4, 0, 1)  # A10*B00 -> C10
    add_term(4, 1, 1)  # A10*B01 -> C10
    add_term(0, 0, 1, -1)  # -A00*B00 -> C10
    add_term(0, 1, 1, -1)  # -A00*B01 -> C10
    
    # M7: (A01 - A11) * (B10 + B11)
    add_term(1, 7, 4)  # A01*B10 -> C11
    add_term(1, 8, 4)  # A01*B11 -> C11
    add_term(5, 7, 4, -1)  # -A11*B10 -> C11
    add_term(5, 8, 4, -1)  # -A11*B11 -> C11
    
    # Now we need to combine these into the C entries.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 - M4 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Wait, I've been adding terms directly to U, V, W with the correct C output rows.
    # But the Strassen algorithm computes intermediate products M1..M7, and then combines them.
    # In the tensor decomposition, each rank-1 term corresponds to one "multiplication" in the algorithm.
    # The issue is that I'm not correctly accounting for the combination step.
    
    # Let me reconsider. In the tensor decomposition, each rank-1 term is:
    # U[a][c] * V[b][c] * W[ab][c]
    # where a indexes A, b indexes B, and ab indexes C.
    
    # For schoolbook: U[i*m+j][c]=1, V[j*p+k][c]=1, W[k*n+i][c]=1
    # This computes C[i][k] += A[i][j] * B[j][k]
    
    # For Strassen, the intermediate products M1..M7 are linear combinations of A and B entries.
    # Each M is a product of a linear combination of A entries and a linear combination of B entries.
    # In the tensor decomposition, each such product corresponds to multiple rank-1 terms.
    
    # Actually, the standard way to decompose Strassen is to note that each M is a bilinear form,
    # and each bilinear form can be written as a sum of rank-1 tensors.
    
    # M1 = A00 * (B00 + B11) = A00*B00 + A00*B11
    # This is two rank-1 terms, which I've already added.
    
    # But then the combination step (C00 = M1 + M4 - M5 + M7) is a linear combination of the M's.
    # In the tensor decomposition, this is handled by the W matrix.
    
    # Let me re-examine. The W matrix should have the coefficients for the combination.
    
    # Actually, I think I need to be more careful. Let me define the decomposition properly.
    
    # Each rank-1 term in the decomposition corresponds to a specific (u_row, v_row, w_row, coefficient).
    # The u_row corresponds to an entry of A, v_row to an entry of B, and w_row to an entry of C.
    
    # For M1 = A00 * (B00 + B11):
    # This contributes to C00.
    # The terms are:
    # 1. A00 * B00 -> C00: u=0, v=0, w=0, coeff=1
    # 2. A00 * B11 -> C00: u=0, v=8, w=0, coeff=1
    
    # For M2 = (A00 + A11) * B10:
    # This contributes to C10.
    # The terms are:
    # 1. A00 * B10 -> C10: u=0, v=7, w=1, coeff=1
    # 2. A11 * B10 -> C10: u=5, v=7, w=1, coeff=1
    
    # And so on.
    
    # But the combination step means that the final C entries are linear combinations of the M's.
    # In the tensor decomposition, this is already accounted for by the fact that each M contributes
    # to specific C entries with specific coefficients.
    
    # I think the decomposition I've built so far is correct for the Strassen part.
    # Let me continue with the schoolbook part for the remaining 5 columns of B_11.
    
    # Schoolbook for A_11 (2x2) * B_11[:, 2:7] (2x5)
    # For i=0,1 (rows of A_11), j=0,1 (cols of A_11), k=2..6 (cols of B_11):
    #   u_row = i*4 + j
    #   v_row = (0 or 1)*7 + k  (since B_11 is rows 0,1 of B)
    #   w_row = k*3 + i
    
    for i in range(2):
        for j in range(2):
            for k in range(2, 7):
                u_row = i * 4 + j
                v_row = k  # B row 0, col k
                w_row = k * 3 + i
                add_term(u_row, v_row, w_row)
                u_row2 = i * 4 + j
                v_row2 = 7 + k  # B row 1, col k
                w_row2 = k * 3 + i
                add_term(u_row2, v_row2, w_row2)
    
    # col is now 28 + 14 + 20 = 62? Let me count:
    # Strassen part: 2+2+2+2+2+4+4 = 18 terms
    # Schoolbook part: 2*2*5 = 20 terms
    # Total for A_11*B_11: 38 terms
    # But I expected 27. Something's wrong.
    
    # The issue is that I'm treating each term in the Strassen expansion as a separate rank-1 term,
    # but that's not how it works. The Strassen algorithm uses 7 multiplications (M1..M7),
    # and each multiplication is a product of two linear combinations.
    
    # In the tensor decomposition, each multiplication corresponds to a rank-1 tensor where
    # the U and V matrices have linear combinations (not just single entries).
    
    # But the problem requires U, V, W to have entries that are 0, 1, or -1 (or half-integers).
    # So each rank-1 term must correspond to a single entry of A times a single entry of B.
    
    # This means that the Strassen decomposition, when expanded into rank-1 terms, uses more than 7 terms.
    
    # Let me reconsider the approach. Maybe I should just use schoolbook for everything,
    # which gives rank 84, and see if I can find a better decomposition.
    
    # Actually, let me restart with a cleaner approach.
    
    # For now, let me just return the schoolbook decomposition to ensure correctness.
    # I can try to optimize later.
    
    # Reset
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook for all of A (3x4) * B (4x7)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # This gives rank 84, which is the schoolbook rank.
    # Let me try to implement a better decomposition.
    
    # For now, return schoolbook.
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
