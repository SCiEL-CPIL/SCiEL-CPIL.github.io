"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 150  # seconds
    
    n, m, p = DIMENSIONS
    
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    # We don't know rank yet, so we'll build lists of tuples (index, value)
    # and then convert to dense matrices at the end.
    # Or we can estimate rank and pre-allocate.
    # Estimated rank: 98.
    
    U_entries = []
    V_entries = []
    W_entries = []
    
    # Helper to add a term
    # term_u: list of (row_index, coeff)
    # term_v: list of (row_index, coeff)
    # term_w: list of (row_index, coeff)
    def add_term(u_idx_coeffs, v_idx_coeffs, w_idx_coeffs):
        U_entries.append(u_idx_coeffs)
        V_entries.append(v_idx_coeffs)
        W_entries.append(w_idx_coeffs)
        
    def schoolbook_block(a_row_start, a_col_start, a_rows, a_cols, 
                         b_row_start, b_col_start, b_rows, b_cols,
                         c_row_start, c_col_start, c_rows, c_cols):
        """
        Adds schoolbook decomposition for a block multiplication.
        A block: a_rows x a_cols
        B block: b_rows x b_cols (a_cols must equal b_rows)
        C block: c_rows x c_cols
        """
        # Verify dimensions
        if a_cols != b_rows:
            return
        if a_rows != c_rows or b_cols != c_cols:
            return
            
        for i in range(a_rows):
            for j in range(a_cols):
                for k in range(b_cols):
                    # Global indices
                    u_row = (a_row_start + i) * m + (a_col_start + j)
                    v_row = (b_row_start + j) * p + (b_col_start + k)
                    w_row = (c_col_start + k) * n + (c_row_start + i)
                    
                    add_term([(u_row, 1)], [(v_row, 1)], [(w_row, 1)])
                    
    def strassen_block(a_row_start, a_col_start, 
                       b_row_start, b_col_start,
                       c_row_start, c_col_start):
        """
        Adds Strassen decomposition for a 2x2x2 block multiplication.
        Assumes a_rows=2, a_cols=2, b_rows=2, b_cols=2, c_rows=2, c_cols=2.
        """
        # Local indices 0,1
        # A00, A01, A10, A11
        # B00, B01, B10, B11
        # C00, C01, C10, C11
        
        # Term 1: (A00 + A11) * (B00 + B11)
        # C00 += 1, C11 += 1
        add_term(
            [((a_row_start + 0) * m + (a_col_start + 0), 1), 
             ((a_row_start + 1) * m + (a_col_start + 1), 1)],
            [((b_row_start + 0) * p + (b_col_start + 0), 1), 
             ((b_row_start + 1) * p + (b_col_start + 1), 1)],
            [((c_col_start + 0) * n + (c_row_start + 0), 1), 
             ((c_col_start + 1) * n + (c_row_start + 1), 1)]
        )
        
        # Term 2: (A10 + A11) * B00
        # C10 += 1, C11 -= 1
        add_term(
            [((a_row_start + 1) * m + (a_col_start + 0), 1), 
             ((a_row_start + 1) * m + (a_col_start + 1), 1)],
            [((b_row_start + 0) * p + (b_col_start + 0), 1)],
            [((c_col_start + 0) * n + (c_row_start + 1), 1), 
             ((c_col_start + 1) * n + (c_row_start + 1), -1)]
        )
        
        # Term 3: A00 * (B10 - B11)
        # C01 += 1, C11 += 1
        add_term(
            [((a_row_start + 0) * m + (a_col_start + 0), 1)],
            [((b_row_start + 1) * p + (b_col_start + 0), 1), 
             ((b_row_start + 1) * p + (b_col_start + 1), -1)],
            [((c_col_start + 1) * n + (c_row_start + 0), 1), 
             ((c_col_start + 1) * n + (c_row_start + 1), 1)]
        )
        
        # Term 4: A11 * (B01 - B00)
        # C00 += 1, C10 += 1
        add_term(
            [((a_row_start + 1) * m + (a_col_start + 1), 1)],
            [((b_row_start + 0) * p + (b_col_start + 1), 1), 
             ((b_row_start + 0) * p + (b_col_start + 0), -1)],
            [((c_col_start + 0) * n + (c_row_start + 0), 1), 
             ((c_col_start + 0) * n + (c_row_start + 1), 1)]
        )
        
        # Term 5: (A00 + A10) * B11
        # C00 -= 1, C01 += 1
        add_term(
            [((a_row_start + 0) * m + (a_col_start + 0), 1), 
             ((a_row_start + 1) * m + (a_col_start + 0), 1)],
            [((b_row_start + 1) * p + (b_col_start + 1), 1)],
            [((c_col_start + 0) * n + (c_row_start + 0), -1), 
             ((c_col_start + 1) * n + (c_row_start + 0), 1)]
        )
        
        # Term 6: (A10 - A00) * (B00 + B01)
        # C11 += 1
        add_term(
            [((a_row_start + 1) * m + (a_col_start + 0), 1), 
             ((a_row_start + 0) * m + (a_col_start + 0), -1)],
            [((b_row_start + 0) * p + (b_col_start + 0), 1), 
             ((b_row_start + 0) * p + (b_col_start + 1), 1)],
            [((c_col_start + 1) * n + (c_row_start + 1), 1)]
        )
        
        # Term 7: (A01 - A11) * (B10 + B11)
        # C00 += 1
        add_term(
            [((a_row_start + 0) * m + (a_col_start + 1), 1), 
             ((a_row_start + 1) * m + (a_col_start + 1), -1)],
            [((b_row_start + 1) * p + (b_col_start + 0), 1), 
             ((b_row_start + 1) * p + (b_col_start + 1), 1)],
            [((c_col_start + 0) * n + (c_row_start + 0), 1)]
        )

    # Block partition:
    # n=4: rows 0-1, 2-3
    # m=5: cols 0-2, 3-4  (splits 3, 2)
    # p=5: cols 0-2, 3-4  (splits 3, 2)
    
    # A blocks:
    # A00: rows 0-1, cols 0-2
    # A01: rows 0-1, cols 3-4
    # A10: rows 2-3, cols 0-2
    # A11: rows 2-3, cols 3-4
    
    # B blocks:
    # B00: rows 0-2, cols 0-2
    # B01: rows 0-2, cols 3-4
    # B10: rows 3-4, cols 0-2
    # B11: rows 3-4, cols 3-4
    
    # C blocks:
    # C00: rows 0-1, cols 0-2
    # C01: rows 0-1, cols 3-4
    # C10: rows 2-3, cols 0-2
    # C11: rows 2-3, cols 3-4
    
    # C00 = A00 B00 + A01 B10
    # A00 B00: 2x3 x 3x3 -> Schoolbook
    schoolbook_block(0, 0, 2, 3, 0, 0, 3, 3, 0, 0, 2, 3)
    # A01 B10: 2x2 x 2x3 -> Schoolbook
    schoolbook_block(0, 3, 2, 2, 3, 0, 2, 3, 0, 0, 2, 3)
    
    # C01 = A00 B01 + A01 B11
    # A00 B01: 2x3 x 3x2 -> Schoolbook
    schoolbook_block(0, 0, 2, 3, 0, 3, 3, 2, 0, 3, 2, 2)
    # A01 B11: 2x2 x 2x2 -> Strassen
    strassen_block(0, 3, 3, 3, 0, 3)
    
    # C10 = A10 B00 + A11 B10
    # A10 B00: 2x3 x 3x3 -> Schoolbook
    schoolbook_block(2, 0, 2, 3, 0, 0, 3, 3, 2, 0, 2, 3)
    # A11 B10: 2x2 x 2x3 -> Schoolbook
    schoolbook_block(2, 3, 2, 2, 3, 0, 2, 3, 2, 0, 2, 3)
    
    # C11 = A10 B01 + A11 B11
    # A10 B01: 2x3 x 3x2 -> Schoolbook
    schoolbook_block(2, 0, 2, 3, 0, 3, 3, 2, 2, 3, 2, 2)
    # A11 B11: 2x2 x 2x2 -> Strassen
    strassen_block(2, 3, 3, 3, 2, 3)
    
    rank = len(U_entries)
    
    # Construct dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for u_idx, val in U_entries[col]:
            U[u_idx][col] = val
        for v_idx, val in V_entries[col]:
            V[v_idx][col] = val
        for w_idx, val in W_entries[col]:
            W[w_idx][col] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
