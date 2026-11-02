"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Matrix multiplication: A (2x4) * B (4x5) = C (2x5)
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank), W[k*n+i, r]
    #
    # Strategy: Split B into B_left (4x2) and B_right (4x3)
    # C[:, 0:2] = A * B_left, C[:, 2:5] = A * B_right
    # For each part, split A into A_left (2x2) and A_right (2x2)
    # A_left * B_sub (2x2 by 2xk) and A_right * B_sub (2x2 by 2xk)
    # Use Strassen (rank 7) for 2x2 by 2x2, schoolbook (rank 4) for 2x2 by 2x1
    #
    # B_left (4x2): split rows into top (2x2) and bottom (2x2)
    #   A_left * B_left_top (2x2 by 2x2): rank 7
    #   A_right * B_left_bot (2x2 by 2x2): rank 7
    # B_right (4x3): split rows into top (2x3) and bottom (2x3)
    #   A_left * B_right_top (2x2 by 2x3): rank 11 (7 for 2x2 part + 4 for 2x1 part)
    #   A_right * B_right_bot (2x2 by 2x3): rank 11
    # Total: 7 + 7 + 11 + 11 = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        """Add Strassen decomposition for 2x2 * 2x2 = 2x2"""
        nonlocal col
        # A indices in U: (a_row_off + i) * m + (a_col_off + j)
        # B indices in V: (b_row_off + j) * p + (b_col_off + k)
        # C indices in W: (c_col_off + k) * n + (c_row_off + i)
        
        u00 = (a_row_off + 0) * m + (a_col_off + 0)
        u01 = (a_row_off + 0) * m + (a_col_off + 1)
        u10 = (a_row_off + 1) * m + (a_col_off + 0)
        u11 = (a_row_off + 1) * m + (a_col_off + 1)
        
        v00 = (b_row_off + 0) * p + (b_col_off + 0)
        v01 = (b_row_off + 0) * p + (b_col_off + 1)
        v10 = (b_row_off + 1) * p + (b_col_off + 0)
        v11 = (b_row_off + 1) * p + (b_col_off + 1)
        
        w00 = (c_col_off + 0) * n + (c_row_off + 0)
        w01 = (c_col_off + 1) * n + (c_row_off + 0)
        w10 = (c_col_off + 0) * n + (c_row_off + 1)
        w11 = (c_col_off + 1) * n + (c_row_off + 1)
        
        # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
        U[u00][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00 -> C10 += M2, C11 -= M2
        U[u10][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        W[w11][col] -= 1
        col += 1
        
        # M3 = a00(b01-b11) -> C01 += M3, C11 += M3
        U[u00][col] += 1
        V[v01][col] += 1
        V[v11][col] -= 1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M4 = a11(b10-b00) -> C00 += M4, C10 += M4
        U[u11][col] += 1
        V[v10][col] += 1
        V[v00][col] -= 1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11 -> C00 -= M5, C01 += M5
        U[u00][col] += 1
        U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1
        W[w01][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01) -> C11 += M6
        U[u10][col] += 1
        U[u00][col] -= 1
        V[v00][col] += 1
        V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11) -> C00 += M7
        U[u01][col] += 1
        U[u11][col] -= 1
        V[v10][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        col += 1
    
    def add_schoolbook_2x2x2x1(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        """Add schoolbook for 2x2 * 2x1 = 2x1"""
        nonlocal col
        u00 = (a_row_off + 0) * m + (a_col_off + 0)
        u01 = (a_row_off + 0) * m + (a_col_off + 1)
        u10 = (a_row_off + 1) * m + (a_col_off + 0)
        u11 = (a_row_off + 1) * m + (a_col_off + 1)
        
        v0 = (b_row_off + 0) * p + b_col_off
        v1 = (b_row_off + 1) * p + b_col_off
        
        w0 = (c_col_off) * n + (c_row_off + 0)
        w1 = (c_col_off) * n + (c_row_off + 1)
        
        # C0 += a00*b0
        U[u00][col] = 1
        V[v0][col] = 1
        W[w0][col] = 1
        col += 1
        
        # C0 += a01*b1
        U[u01][col] = 1
        V[v1][col] = 1
        W[w0][col] = 1
        col += 1
        
        # C1 += a10*b0
        U[u10][col] = 1
        V[v0][col] = 1
        W[w1][col] = 1
        col += 1
        
        # C1 += a11*b1
        U[u11][col] = 1
        V[v1][col] = 1
        W[w1][col] = 1
        col += 1
    
    # Block 1: A_left (rows 0-1, cols 0-1) * B_left_top (rows 0-1, cols 0-1) -> C[:, 0-1]
    # a_row_off=0, a_col_off=0, b_row_off=0, b_col_off=0, c_row_off=0, c_col_off=0
    add_strassen_2x2x2x2(0, 0, 0, 0, 0, 0)
    
    # Block 2: A_right (rows 0-1, cols 2-3) * B_left_bot (rows 2-3, cols 0-1) -> C[:, 0-1]
    # a_row_off=0, a_col_off=2, b_row_off=2, b_col_off=0, c_row_off=0, c_col_off=0
    add_strassen_2x2x2x2(0, 2, 2, 0, 0, 0)
    
    # Block 3: A_left (rows 0-1, cols 0-1) * B_right_top (rows 0-1, cols 2-4) -> C[:, 2-4]
    # B_right_top is 2x3: split into cols 2-3 (2x2) and col 4 (2x1)
    # 3a: 2x2 * 2x2 (cols 2-3)
    add_strassen_2x2x2x2(0, 0, 0, 2, 0, 2)
    # 3b: 2x2 * 2x1 (col 4)
    add_schoolbook_2x2x2x1(0, 0, 0, 4, 0, 4)
    
    # Block 4: A_right (rows 0-1, cols 2-3) * B_right_bot (rows 2-3, cols 2-4) -> C[:, 2-4]
    # 4a: 2x2 * 2x2 (cols 2-3)
    add_strassen_2x2x2x2(0, 2, 2, 2, 0, 2)
    # 4b: 2x2 * 2x1 (col 4)
    add_schoolbook_2x2x2x1(0, 2, 2, 4, 0, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
