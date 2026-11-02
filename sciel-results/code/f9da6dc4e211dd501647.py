"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        # a_rows, a_cols, b_rows, b_cols, c_rows, c_cols are lists of 2 indices
        a0_idx = a_rows[0] * m + a_cols[0]
        a1_idx = a_rows[0] * m + a_cols[1]
        a2_idx = a_rows[1] * m + a_cols[0]
        a3_idx = a_rows[1] * m + a_cols[1]
        
        b0_idx = b_rows[0] * p + b_cols[0]
        b1_idx = b_rows[0] * p + b_cols[1]
        b2_idx = b_rows[1] * p + b_cols[0]
        b3_idx = b_rows[1] * p + b_cols[1]
        
        c00_idx = c_cols[0] * n + c_rows[0]
        c01_idx = c_cols[1] * n + c_rows[0]
        c10_idx = c_cols[0] * n + c_rows[1]
        c11_idx = c_cols[1] * n + c_rows[1]
        
        # M1 = (a0+a3)(b0+b3)
        U[a0_idx][col] += 1
        U[a3_idx][col] += 1
        V[b0_idx][col] += 1
        V[b3_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M2 = (a2+a3)b0
        U[a2_idx][col] += 1
        U[a3_idx][col] += 1
        V[b0_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a0(b1-b2)
        U[a0_idx][col] += 1
        V[b1_idx][col] += 1
        V[b2_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M4 = a3(b2-b0)
        U[a3_idx][col] += 1
        V[b2_idx][col] += 1
        V[b0_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a0+a1)b3
        U[a0_idx][col] += 1
        U[a1_idx][col] += 1
        V[b3_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        # M6 = (a2-a0)(b0+b1)
        U[a2_idx][col] += 1
        U[a0_idx][col] -= 1
        V[b0_idx][col] += 1
        V[b1_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (a1-a3)(b2+b3)
        U[a1_idx][col] += 1
        U[a3_idx][col] -= 1
        V[b2_idx][col] += 1
        V[b3_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_dot2_block(a_row, a_cols, b_rows, b_col, c_row, c_col):
        nonlocal col
        a0_idx = a_row * m + a_cols[0]
        a1_idx = a_row * m + a_cols[1]
        
        b0_idx = b_rows[0] * p + b_col
        b1_idx = b_rows[1] * p + b_col
        
        c_idx = c_col * n + c_row
        
        # M1 = a0 * b0
        U[a0_idx][col] += 1
        V[b0_idx][col] += 1
        W[c_idx][col] += 1
        col += 1
        
        # M2 = a1 * b1
        U[a1_idx][col] += 1
        V[b1_idx][col] += 1
        W[c_idx][col] += 1
        col += 1

    # A1 B part (rows 0,1 of A)
    # A11 (cols 0,1) * B rows 0,1
    # B rows 0,1 split into cols 0-3 and 4-7
    # Cols 0-3 split into 0-1 and 2-3
    # Cols 4-7 split into 4-5 and 6-7
    
    # Block 1: A11 * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    # Block 2: A11 * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    # Block 3: A11 * B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen_block([0,1], [0,1], [0,1], [4,5], [0,1], [4,5])
    # Block 4: A11 * B[0:2, 6:8] -> C[0:2, 6:8]
    add_strassen_block([0,1], [0,1], [0,1], [6,7], [0,1], [6,7])
    
    # A12 (cols 2,3) * B rows 2,3
    # Block 5: A12 * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    # Block 6: A12 * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    # Block 7: A12 * B[2:4, 4:6] -> C[0:2, 4:6]
    add_strassen_block([0,1], [2,3], [2,3], [4,5], [0,1], [4,5])
    # Block 8: A12 * B[2:4, 6:8] -> C[0:2, 6:8]
    add_strassen_block([0,1], [2,3], [2,3], [6,7], [0,1], [6,7])
    
    # A2 B part (row 2 of A)
    # A21 (cols 0,1) * B rows 0,1
    for k in range(4):
        add_dot2_block(2, [0,1], [0,1], k, 2, k)
        
    # A22 (cols 2,3) * B rows 2,3
    for k in range(4, 8):
        add_dot2_block(2, [2,3], [2,3], k, 2, k)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
