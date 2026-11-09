def construct():
    n, m, p = 4, 5, 5
    rank = 96
    
    U = [[0] * rank for _ in range(n * m)]  # 20 x 96
    V = [[0] * rank for _ in range(m * p)]  # 25 x 96
    W = [[0] * rank for _ in range(p * n)]  # 20 x 96
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_entries, v_entries, w_entries):
        nonlocal col
        for (row, val) in u_entries:
            U[row][col] += val
        for (row, val) in v_entries:
            V[row][col] += val
        for (row, val) in w_entries:
            W[row][col] += val
        col += 1
    
    # Strassen for 2x2x2: 
    # A indices: (ai, aj) where ai is row in {0,1}, aj is col in {0,1}
    # B indices: (bj, bk) where bj is row in {0,1}, bk is col in {0,1}
    # C indices: (ci, ck) where ci is row in {0,1}, ck is col in {0,1}
    # 
    # U row: (ai_global * m + aj_global)
    # V row: (bj_global * p + bk_global)
    # W row: (ck_global * n + ci_global)
    
    def add_strassen(a_row_offset, b_col_offset, j_offset):
        """Add Strassen decomposition for 2x2x2 at given offsets.
        a_row_offset: global row offset for A (0 or 2)
        b_col_offset: global col offset for B (0, 2, or 4)
        j_offset: global middle index offset (0 or 2)
        """
        # A rows: a_row_offset, a_row_offset+1
        # A cols (j): j_offset, j_offset+1
        # B rows (j): j_offset, j_offset+1
        # B cols: b_col_offset, b_col_offset+1
        # C rows: a_row_offset, a_row_offset+1
        # C cols: b_col_offset, b_col_offset+1
        
        # M1 = (a00+a11)(b00+b11)
        add_term(
            [(a_row_offset * m + j_offset, 1), ((a_row_offset+1) * m + (j_offset+1), 1)],
            [(j_offset * p + b_col_offset, 1), ((j_offset+1) * p + (b_col_offset+1), 1)],
            [(b_col_offset * n + a_row_offset, 1), ((b_col_offset+1) * n + (a_row_offset+1), 1)]
        )
        
        # M2 = (a10+a11)*b00
        add_term(
            [((a_row_offset+1) * m + j_offset, 1), ((a_row_offset+1) * m + (j_offset+1), 1)],
            [(j_offset * p + b_col_offset, 1)],
            [(b_col_offset * n + (a_row_offset+1), 1), ((b_col_offset+1) * n + (a_row_offset+1), -1)]
        )
        
        # M3 = a00*(b01-b11)
        add_term(
            [(a_row_offset * m + j_offset, 1)],
            [(j_offset * p + (b_col_offset+1), 1), ((j_offset+1) * p + (b_col_offset+1), -1)],
            [((b_col_offset+1) * n + a_row_offset, 1), ((b_col_offset+1) * n + (a_row_offset+1), 1)]
        )
        
        # M4 = a11*(b10-b00)
        add_term(
            [((a_row_offset+1) * m + (j_offset+1), 1)],
            [((j_offset+1) * p + b_col_offset, 1), (j_offset * p + b_col_offset, -1)],
            [(b_col_offset * n + a_row_offset, 1), (b_col_offset * n + (a_row_offset+1), 1)]
        )
        
        # M5 = (a00+a01)*b11
        add_term(
            [(a_row_offset * m + j_offset, 1), (a_row_offset * m + (j_offset+1), 1)],
            [((j_offset+1) * p + (b_col_offset+1), 1)],
            [(b_col_offset * n + a_row_offset, -1), ((b_col_offset+1) * n + a_row_offset, 1)]
        )
        
        # M6 = (a10-a00)*(b00+b01)
        add_term(
            [((a_row_offset+1) * m + j_offset, 1), (a_row_offset * m + j_offset, -1)],
            [(j_offset * p + b_col_offset, 1), (j_offset * p + (b_col_offset+1), 1)],
            [((b_col_offset+1) * n + (a_row_offset+1), 1)]
        )
        
        # M7 = (a01-a11)*(b10+b11)
        add_term(
            [(a_row_offset * m + (j_offset+1), 1), ((a_row_offset+1) * m + (j_offset+1), -1)],
            [((j_offset+1) * p + b_col_offset, 1), ((j_offset+1) * p + (b_col_offset+1), 1)],
            [(b_col_offset * n + a_row_offset, 1), (b_col_offset * n + (a_row_offset+1), -1)]
        )
    
    # Schoolbook for 2x3x2:
    def add_schoolbook_2x3x2(a_row_offset, b_col_offset, j_start, j_count=3):
        """Schoolbook for A(2x3) * B(3x2) → C(2x2)"""
        for ai in range(2):
            for j in range(j_count):
                for bk in range(2):
                    add_term(
                        [((a_row_offset + ai) * m + (j_start + j), 1)],
                        [((j_start + j) * p + (b_col_offset + bk), 1)],
                        [((b_col_offset + bk) * n + (a_row_offset + ai), 1)]
                    )
    
    # Schoolbook for 2x5x1:
    def add_schoolbook_2x5x1(a_row_offset, b_col, j_start=0, j_count=5):
        """Schoolbook for A(2x5) * B(5x1) → C(2x1)"""
        for ai in range(2):
            for j in range(j_count):
                add_term(
                    [((a_row_offset + ai) * m + (j_start + j), 1)],
                    [((j_start + j) * p + b_col, 1)],
                    [(b_col * n + (a_row_offset + ai), 1)]
                )
    
    # Sub-problem 1: A1*B1 (rows 0-1, cols 0-1)
    add_strassen(a_row_offset=0, b_col_offset=0, j_offset=0)  # 2x2x2 for j=0,1
    add_schoolbook_2x3x2(a_row_offset=0, b_col_offset=0, j_start=2)  # 2x3x2 for j=2,3,4
    
    # Sub-problem 2: A2*B1 (rows 2-3, cols 0-1)
    add_strassen(a_row_offset=2, b_col_offset=0, j_offset=0)
    add_schoolbook_2x3x2(a_row_offset=2, b_col_offset=0, j_start=2)
    
    # Sub-problem 3: A1*B2 (rows 0-1, cols 2-3)
    add_strassen(a_row_offset=0, b_col_offset=2, j_offset=0)
    add_schoolbook_2x3x2(a_row_offset=0, b_col_offset=2, j_start=2)
    
    # Sub-problem 4: A2*B2 (rows 2-3, cols 2-3)
    add_strassen(a_row_offset=2, b_col_offset=2, j_offset=0)
    add_schoolbook_2x3x2(a_row_offset=2, b_col_offset=2, j_start=2)
    
    # Sub-problem 5: A1*B3 (rows 0-1, col 4)
    add_schoolbook_2x5x1(a_row_offset=0, b_col=4)
    
    # Sub-problem 6: A2*B3 (rows 2-3, col 4)
    add_schoolbook_2x5x1(a_row_offset=2, b_col=4)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
