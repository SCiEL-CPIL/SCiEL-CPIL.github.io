import time

def construct():
    """
    Returns a rank-42 exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    
    The decomposition uses Strassen's algorithm applied to 2x2 blocks, with the 4x6 matrix
    split into three 2x6 blocks (columns 0-1, 2-3, 4-5). Each 2x2 x 2x6 block is computed
    using 7 Strassen multiplications, giving 7 * 6 = 42 total multiplications.
    
    U: (n*m, rank) = (8, 42)
    V: (m*p, rank) = (24, 42) 
    W: (p*n, rank) = (12, 42)
    
    W indexing convention: W[k*n + i, r] where k is output column, i is output row.
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen terms for 2x2 x 2x2 multiplication
    # Each term: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: list of (row, col, coeff) for the 2x2 A block
    # b_coeffs: list of (row, col, coeff) for the 2x2 B block  
    # c_coeffs: list of (row, col, coeff) for the 2x2 C block contribution
    strassen_terms = [
        # M1: (A00 + A11) * (B00 + B11) -> C00 += M1, C11 += M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A10 + A11) * B00 -> C10 += M2, C11 -= M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3: A00 * (B01 - B11) -> C01 += M3, C11 += M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4: A11 * (B10 - B00) -> C00 += M4, C10 += M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5: (A00 + A01) * B11 -> C00 -= M5, C01 += M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6: (A10 - A00) * (B00 + B01) -> C11 += M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7: (A01 - A11) * (B10 + B11) -> C00 += M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    # Define the 6 blocks: (a_row_offset, a_col_offset, b_row_offset, b_col_offset)
    # A is 2x4, split into 2 row-blocks (offset 0, 2 in cols) 
    # B is 4x6, split into 2 row-blocks (offset 0, 2) and 3 col-blocks (offset 0, 2, 4)
    blocks = [
        (0, 0, 0, 0),   # A rows 0-1, cols 0-1; B rows 0-1, cols 0-1
        (0, 0, 0, 2),   # A rows 0-1, cols 0-1; B rows 0-1, cols 2-3
        (0, 0, 0, 4),   # A rows 0-1, cols 0-1; B rows 0-1, cols 4-5
        (0, 2, 2, 0),   # A rows 0-1, cols 2-3; B rows 2-3, cols 0-1
        (0, 2, 2, 2),   # A rows 0-1, cols 2-3; B rows 2-3, cols 2-3
        (0, 2, 2, 4)    # A rows 0-1, cols 2-3; B rows 2-3, cols 4-5
    ]
    
    col = 0
    for a_r_off, a_c_off, b_r_off, b_c_off in blocks:
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            # Fill U: U index = (a_row * m + a_col)
            for i, j, coeff in a_coeffs:
                u_idx = i * m + (j + a_c_off)
                U[u_idx][col] += coeff
            
            # Fill V: V index = (b_row * p + b_col)
            for i, j, coeff in b_coeffs:
                v_idx = (i + b_r_off) * p + (j + b_c_off)
                V[v_idx][col] += coeff
            
            # Fill W: W index = k * n + i, where k is output col, i is output row
            # Output C has rows 0-1 (from a_r_off which is always 0 here), cols from b_c_off
            for i, j, coeff in c_coeffs:
                global_i = i  # output row (0 or 1)
                global_k = j + b_c_off  # output column
                w_idx = global_k * n + global_i
                W[w_idx][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
