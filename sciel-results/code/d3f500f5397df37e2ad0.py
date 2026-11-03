"""Program for matrix multiplication 2x4 by 4x7 with rank-minimal tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    rank = 46
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper: add a 2x2x2 Strassen block
    # For multiplying A (2x2) by B (2x2), Strassen uses 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(A_rows, B_cols, C_rows, C_cols):
        nonlocal col
        # A_rows: list of 2 row indices in U (for the 2x2 A submatrix)
        # B_cols: list of 2 col indices in V (for the 2x2 B submatrix)
        # C_rows: list of 2 row indices in W (for the 2x2 C submatrix)
        # C_cols: list of 2 col indices in W (for the 2x2 C submatrix)
        # W index convention: W[k*n+i, r] where k is row of B (0..p-1), i is col of A (0..n-1)
        # So W index = k*n + i
        
        # The 7 Strassen terms
        terms = [
            # (M1, coeff_A00, coeff_A01, coeff_A10, coeff_A11, coeff_B00, coeff_B01, coeff_B10, coeff_B11, C00, C01, C10, C11)
            (1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0),
            (2, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0),
            (3, 1, 0, 0, 0, 0, 0, 1, -1, 0, 1, 0, 0),
            (4, 0, 0, 0, 1, 0, 1, -1, 0, 1, 0, 0, 0),
            (5, 1, 1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0),
            (6, -1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1),
            (7, 0, 1, 0, -1, 0, 1, 0, 1, 1, 0, 0, 0),
        ]
        
        for (coeff_M, cA00, cA01, cA10, cA11, cB00, cB01, cB10, cB11, cC00, cC01, cC10, cC11) in terms:
            # A part: A is 2x2, rows 0,1 in A_rows
            if cA00 != 0:
                U[A_rows[0], col] += cA00
            if cA01 != 0:
                U[A_rows[0], col] += cA01  # Wait, A01 is row 0, col 1
            # Let me reconsider. A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
            # A_rows[0] corresponds to row 0 of the A submatrix, A_rows[1] to row 1
            # But U index is i*m+j, where i is row of A, j is col of A (within the 4 columns)
            # Actually, for a 2x2 block of A, the rows are fixed (say rows i1, i1+1) and cols are j1, j1+1
            # U[i1*m+j1, col] for A[0,0]
            # U[i1*m+j1+1, col] for A[0,1]
            # U[(i1+1)*m+j1, col] for A[1,0]
            # U[(i1+1)*m+j1+1, col] for A[1,1]
            
            col += 1
    
    # Let me redo this more carefully.
    # For 2x4 x 4x7:
    # A is 2x4, B is 4x7
    # Split A into two 2x2 blocks: A_left (cols 0,1) and A_right (cols 2,3)
    # Split B into: B_top (rows 0,1) and B_bottom (rows 2,3), each 2x7
    # Then C = A_left * B_top + A_right * B_bottom
    # Each product is 2x7
    # For each 2x7 product, split the 7 columns into 2+2+3 or similar
    # Let's do: 2x2 + 2x2 + 2x3
    # 2x2: rank 7 (Strassen)
    # 2x2: rank 7
    # 2x3: need rank? 2x3x2 is rank 5? No, 2x2x3 is rank 10? 
    # Actually, the rank of the 2x2x3 tensor is 10 (since 2x2x2 is 7, and adding one more column... not additive)
    # Let's use a simpler approach: 2x7 can be done as 7 individual 2x2x1 blocks? No.
    # 2x1: rank 2 (since A is 2x1, B is 1x1, C is 2x1, so we need 2 products: A0*B, A1*B)
    # So 2x7 = 7 * (2x1) would be rank 14.
    # But we can do better by grouping.
    
    # Alternative: use the known result that 2x4x4 has rank 16? Or 15?
    # Let me just use a straightforward approach with some optimization.
    
    # Actually, let me use the following known ranks:
    # 2x2x2: rank 7
    # 2x2x1: rank 2
    # 2x1x1: rank 1
    # For 2x4x7, I'll construct it as:
    # Split the 4 middle dimension into two 2's
    # C = A1 * B1 + A2 * B2 where A1, A2 are 2x2 and B1, B2 are 2x7
    # Each 2x7 product: split 7 into 2+2+3
    # 2x2: rank 7
    # 2x2: rank 7  
    # 2x3: I'll use rank 10 (conservative)
    # Total per 2x7: 7+7+10 = 24
    # Total: 24 * 2 = 48
    
    # Let me implement 2x2x2 Strassen and 2x2x1 and 2x2x3.
    
    def add_2x2x2_strassen(A_row_start, B_col_start, C_row_start, C_col_start):
        nonlocal col
        # A: rows A_row_start, A_row_start+1; cols B_col_start, B_col_start+1 (within A's column space)
        # Wait, A is 2x4. The "rows" of A are 0,1. The "cols" are 0,1,2,3.
        # For a 2x2 block of A, we pick 2 cols from A.
        # B is 4x7. Rows 0,1,2,3; cols 0,1,2,3,4,5,6.
        # For a 2x2 block of B, we pick 2 rows from B and 2 cols from B.
        # C is 2x7. Rows 0,1; cols 0,1,2,3,4,5,6.
        
        # The 7 Strassen terms for multiplying a 2x2 A by a 2x2 B:
        terms = [
            # (A00, A01, A10, A11, B00, B01, B10, B11, C00, C01, C10, C11)
            (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0),   # M1 = (A00+A11)(B00+B11)
            (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0),   # M2 = (A10+A11)B00
            (1, 0, 0, 0, 0, 0, 1, -1, 0, 1, 0, 0),  # M3 = A00(B10-B11)
            (0, 0, 0, 1, 0, 1, -1, 0, 1, 0, 0, 0),  # M4 = A11(B01-B10)
            (1, 1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0),  # M5 = (A00+A01)B11
            (-1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1),  # M6 = (A10-A00)(B00+B10)
            (0, 1, 0, -1, 0, 1, 0, 1, 1, 0, 0, 0),  # M7 = (A01-A11)(B01+B11)
        ]
        
        for (cA00, cA01, cA10, cA11, cB00, cB01, cB10, cB11, cC00, cC01, cC10, cC11) in terms:
            # A part: A is 2x2, row 0/1, col 0/1
            if cA00 != 0:
                U[A_row_start, B_col_start] = U[A_row_start, B_col_start]  # This is wrong, U index is (row, col) but col is the tensor rank index
            # I'm confusing myself. Let me restart with clearer indexing.
            col += 1
    
    # I'm making this too complicated. Let me use a simpler, correct approach.
    # I'll construct the 46-rank decomposition using known Strassen-like techniques.
    
    # For now, let me use rank 48 (2*24) which is definitely correct, and see if I can optimize.
    # Actually, the task says score is the rank, lower is better. The seed is 56.
    # Any rank < 56 is an improvement.
    
    # Let me use a correct rank 48 decomposition.
    # 2x4 x 4x7 = (2x2 x 2x7) + (2x2 x 2x7)
    # Each 2x2 x 2x7: split 7 cols into 2+2+3
    # 2x2 x 2x2: rank 7
    # 2x2 x 2x2: rank 7
    # 2x2 x 2x3: I'll use rank 14 (7 per column pair, so 2x3 = 3 columns, use 2x2+2x1 = 7+2=9? No)
    # 2x2 x 2x1: rank 2
    # So 2x2 x 2x3 = 2x2 x (2x2 + 2x1) = 7 + 2 = 9? But this assumes disjointness.
    # Let's use rank 10 for 2x2x3 to be safe (known rank is 10? Actually, I think it's 10).
    # So per 2x7: 7 + 7 + 10 = 24
    # Total: 48
    
    # Let me implement this correctly.
    
    col = 0
    
    def add_2x2x2_strassen(A_row, B_col):
        nonlocal col
        # A_row: starting row of A (0 or 1, but A is 2x4 so rows are 0,1)
        # B_col: starting col of the B submatrix (0, 2, or 4 for the 7 cols, but B is 4x7)
        # Wait, B is 4x7. Rows 0-3, cols 0-6.
        # For a 2x2 block of B, we need 2 rows (from 0-3) and 2 cols (from 0-6).
        # A is 2x4. Rows 0-1, cols 0-3.
        # For a 2x2 block of A, we need 2 cols (from 0-3).
        # The product of a 2x2 A block and a 2x2 B block gives a 2x2 C block.
        # C is 2x7. Rows 0-1, cols 0-6.
        
        terms = [
            (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0),
            (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0),
            (1, 0, 0, 0, 0, 0, 1, -1, 0, 1, 0, 0),
            (0, 0, 0, 1, 0, 1, -1, 0, 1, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0),
            (-1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1),
            (0, 1, 0, -1, 0, 1, 0, 1, 1, 0, 0, 0),
        ]
        
        for (cA00, cA01, cA10, cA11, cB00, cB01, cB10, cB11, cC00, cC01, cC10, cC11) in terms:
            # A indices: A is 2x4. For a 2x2 block starting at col A_col_start:
            # A[0, A_col_start] -> U[0*4 + A_col_start, col] = U[A_col_start, col]
            # A[0, A_col_start+1] -> U[A_col_start+1, col]
            # A[1, A_col_start] -> U[4 + A_col_start, col]
            # A[1, A_col_start+1] -> U[4 + A_col_start+1, col]
            
            # B indices: B is 4x7. For a 2x2 block starting at row B_row_start, col B_col_start:
            # B[B_row_start, B_col_start] -> V[B_row_start*7 + B_col_start, col]
            # B[B_row_start, B_col_start+1] -> V[B_row_start*7 + B_col_start+1, col]
            # B[B_row_start+1, B_col_start] -> V[(B_row_start+1)*7 + B_col_start, col]
            # B[B_row_start+1, B_col_start+1] -> V[(B_row_start+1)*7 + B_col_start+1, col]
            
            # C indices: C is 2x7. W[k*n+i, r] where k is row of B (0-3), i is col of A (0-1)
            # Wait, the convention is W[k*n+i, r]. n=2. So W index = k*2 + i.
            # k ranges 0-3 (rows of B), i ranges 0-1 (cols of A).
            # For a 2x2 C block, k is B_row_start, B_row_start+1 and i is A_col_start, A_col_start+1? No.
            # i is the column index of A, which is also the column index of C.
            # So for C[0, A_col_start] (row 0 of C, col A_col_start): W index = B_row_start*2 + A_col_start? No.
            # The C matrix is 2x7. Row 0, col j: W[j*2 + 0, r]? No, W[k*n+i] where k is the "output row" related to B's row, i is related to A's col.
            # Actually, for C = A*B, C[i,k] = sum_j A[i,j] * B[j,k]
            # W[k*n+i, r] means the (k,i) entry of C. k is 0-3 (from B's rows), i is 0-1 (from A's cols).
            # So C[i,k] is at W[k*2 + i, r].
            
            # For a 2x2 C block: k in {B_row_start, B_row_start+1}, i in {A_col_start, A_col_start+1}
            # C00 = C[A_col_start, B_row_start] -> W[B_row_start*2 + A_col_start, col]
            # C01 = C[A_col_start+1, B_row_start] -> W[B_row_start*2 + A_col_start+1, col]
            # C10 = C[A_col_start, B_row_start+1] -> W[(B_row_start+1)*2 + A_col_start, col]
            # C11 = C[A_col_start+1, B_row_start+1] -> W[(B_row_start+1)*2 + A_col_start+1, col]
            
            if cA00 != 0:
                U[A_col_start, col] += cA00
            if cA01 != 0:
                U[A_col_start + 1, col] += cA01
            if cA10 != 0:
                U[4 + A_col_start, col] += cA10
            if cA11 != 0:
                U[4 + A_col_start + 1, col] += cA11
            
            if cB00 != 0:
                V[B_row_start * 7 + B_col_start, col] += cB00
            if cB01 != 0:
                V[B_row_start * 7 + B_col_start + 1, col] += cB01
            if cB10 != 0:
                V[(B_row_start + 1) * 7 + B_col_start, col] += cB10
            if cB11 != 0:
                V[(B_row_start + 1) * 7 + B_col_start + 1, col] += cB11
            
            if cC00 != 0:
                W[B_row_start * 2 + A_col_start, col] += cC00
            if cC01 != 0:
                W[B_row_start * 2 + A_col_start + 1, col] += cC01
            if cC10 != 0:
                W[(B_row_start + 1) * 2 + A_col_start, col] += cC10
            if cC11 != 0:
                W[(B_row_start + 1) * 2 + A_col_start + 1, col] += cC11
            
            col += 1
    
    # This is getting messy. Let me just output rank 56 (the schoolbook) to be safe, since I'm not confident in my construction.
    # Actually, the task says the seed is 56 and I need to beat it. Let me try rank 48.
    
    # I'll use a simpler correct decomposition: rank 48 via 2*(7+7+10) but I'll implement 2x2x3 as two 2x2x1 and one 2x2x1? No.
    # 2x2x1: A is 2x2, B is 2x1, C is 2x1. Rank = 2 (since C[0] = A[0,0]B[0,0] + A[0,1]B[1,0], C[1] = A[1,0]B[0,0] + A[1,1]B[1,0])
    # So for 2x2x3: 3 columns, each needs 2 terms? No, that's rank 6? But we can share the B terms.
    # Actually, 2x2x3: the tensor has rank 10? I'm not sure.
    
    # Let me just use rank 48 and implement it correctly.
    col = 0
    
    # Process two 2x2 A blocks (cols 0-1 and 2-3)
    for A_block in range(2):
        A_col_start = A_block * 2
        # For each 2x7 B block (rows 0-1 and 2-3)
        for B_block in range(2):
            B_row_start = B_block * 2
            # Split 7 cols into 2+2+3
            # First 2 cols: Strassen 2x2x2
            add_2x2x2_strassen(A_col_start, B_row_start)
            # Next 2 cols: Strassen 2x2x2
            add_2x2x2_strassen(A_col_start, B_row_start)  # This is wrong, I need to specify the B col start
            # Last 3 cols: use 3 times 2x2x1 (rank 2 each) = rank 6? Or use a 2x2x3 decomposition.
    
    # I'm clearly struggling with the implementation. Let me just return the schoolbook rank 56 to be safe, since an incorrect decomposition will score 0.
    # The goal is to beat 56, but correctness is paramount.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

