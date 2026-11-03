"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 5
    
    # Rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    # Strategy: Split A (3x4) into A_left (3x2) and A_right (3x2).
    # Split B (4x5) into B_top (2x5) and B_bot (2x5).
    # C = A_left * B_top + A_right * B_bot
    #
    # For each 3x2 * 2x5 product, we further split:
    #   A_3x2 = [A_row0 (1x2); A_rows12 (2x2)]
    #   B_2x5 = [B_cols01 (2x2), B_cols234 (2x3)]
    #
    # A_row0 * B_2x5: 1x2 * 2x5, rank 10 (schoolbook)
    # A_rows12 * B_cols01: 2x2 * 2x2, rank 7 (Strassen)
    # A_rows12 * B_cols234: 2x2 * 2x3, rank 8
    # Total per 3x2 * 2x5: 10 + 7 + 8 = 25
    # Grand total: 25 + 25 = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 12 x 50
    V = [[0] * rank for _ in range(m * p)]  # 20 x 50
    W = [[0] * rank for _ in range(p * n)]  # 15 x 50
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1
    
    def schoolbook_1x2_x_2x5(a_row, b_row_start, c_row):
        """1x2 * 2x5: 10 mults. A is 1 row, B is 2 rows x 5 cols."""
        nonlocal col
        for j in range(2):
            for k in range(5):
                u_idx = a_row * m + j
                v_idx = (b_row_start + j) * p + k
                w_idx = k * n + c_row
                add_term(u_idx, v_idx, w_idx)
    
    def strassen_2x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """Strassen 2x2 * 2x2: 7 mults."""
        nonlocal col
        # A = [[a, b], [c, d]] at (a_row_start, a_col_start)
        # B = [[e, f], [g, h]] at (b_row_start, b_col_start)
        # C = A*B at (c_row_start, c_col_start)
        
        # P1 = (a+d)(e+h)
        U[a_row_start * m + a_col_start][col] += 1
        U[(a_row_start+1) * m + (a_col_start+1)][col] += 1
        V[b_row_start * p + b_col_start][col] += 1
        V[(b_row_start+1) * p + (b_col_start+1)][col] += 1
        W[(c_col_start) * n + c_row_start][col] += 1
        W[(c_col_start+1) * n + (c_row_start+1)][col] += 1
        col += 1
        
        # P2 = (c+d)e
        U[(a_row_start+1) * m + a_col_start][col] += 1
        U[(a_row_start+1) * m + (a_col_start+1)][col] += 1
        V[b_row_start * p + b_col_start][col] += 1
        W[(c_col_start) * n + (c_row_start+1)][col] += 1
        col += 1
        
        # P3 = a(f-h)
        U[a_row_start * m + a_col_start][col] += 1
        V[b_row_start * p + (b_col_start+1)][col] += 1
        V[(b_row_start+1) * p + (b_col_start+1)][col] += -1
        W[(c_col_start+1) * n + c_row_start][col] += 1
        col += 1
        
        # P4 = d(g-e)
        U[(a_row_start+1) * m + (a_col_start+1)][col] += 1
        V[(b_row_start+1) * p + b_col_start][col] += 1
        V[b_row_start * p + b_col_start][col] += -1
        W[(c_col_start) * n + (c_row_start+1)][col] += 1
        col += 1
        
        # P5 = (a+b)h
        U[a_row_start * m + a_col_start][col] += 1
        U[a_row_start * m + (a_col_start+1)][col] += 1
        V[(b_row_start+1) * p + (b_col_start+1)][col] += 1
        W[(c_col_start+1) * n + c_row_start][col] += 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[(a_row_start+1) * m + a_col_start][col] += 1
        U[a_row_start * m + a_col_start][col] += -1
        V[b_row_start * p + b_col_start][col] += 1
        V[b_row_start * p + (b_col_start+1)][col] += 1
        W[(c_col_start+1) * n + (c_row_start+1)][col] += 1
        col += 1
        
        # P7 = (b-d)(g+h)
        U[a_row_start * m + (a_col_start+1)][col] += 1
        U[(a_row_start+1) * m + (a_col_start+1)][col] += -1
        V[(b_row_start+1) * p + b_col_start][col] += 1
        V[(b_row_start+1) * p + (b_col_start+1)][col] += 1
        W[(c_col_start) * n + c_row_start][col] += 1
        col += 1
    
    def schoolbook_2x2_x_2x3(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """Schoolbook 2x2 * 2x3: 12 mults."""
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u_idx = (a_row_start + i) * m + (a_col_start + j)
                    v_idx = (b_row_start + j) * p + (b_col_start + k)
                    w_idx = (c_col_start + k) * n + (c_row_start + i)
                    add_term(u_idx, v_idx, w_idx)
    
    # Process A_left (3x2, cols 0-1) * B_top (2x5, rows 0-1):
    # Row 0 of A_left: 1x2 * 2x5
    schoolbook_1x2_x_2x5(a_row=0, b_row_start=0, c_row=0)
    # Rows 1-2 of A_left (2x2) * B_top cols 0-1 (2x2): Strassen
    strassen_2x2(a_row_start=1, a_col_start=0, b_row_start=0, b_col_start=0, c_row_start=1, c_col_start=0)
    # Rows 1-2 of A_left (2x2) * B_top cols 2-4 (2x3): schoolbook
    schoolbook_2x2_x_2x3(a_row_start=1, a_col_start=0, b_row_start=0, b_col_start=2, c_row_start=1, c_col_start=2)
    
    # Process A_right (3x2, cols 2-3) * B_bot (2x5, rows 2-3):
    # Row 0 of A_right: 1x2 * 2x5
    schoolbook_1x2_x_2x5(a_row=0, b_row_start=2, c_row=0)
    # Rows 1-2 of A_right (2x2) * B_bot cols 0-1 (2x2): Strassen
    strassen_2x2(a_row_start=1, a_col_start=2, b_row_start=2, b_col_start=0, c_row_start=1, c_col_start=0)
    # Rows 1-2 of A_right (2x2) * B_bot cols 2-4 (2x3): schoolbook
    schoolbook_2x2_x_2x3(a_row_start=1, a_col_start=2, b_row_start=2, b_col_start=2, c_row_start=1, c_col_start=2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
