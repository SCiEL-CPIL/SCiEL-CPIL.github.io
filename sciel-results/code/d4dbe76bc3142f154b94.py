"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 8 blocks
    # Block 0: A[:, 0:2], B[0:2, 0:2], C[:, 0:2]
    # Block 1: A[:, 0:2], B[0:2, 2:4], C[:, 2:4]
    # Block 2: A[:, 0:2], B[0:2, 4:6], C[:, 4:6]
    # Block 3: A[:, 0:2], B[0:2, 6:8], C[:, 6:8]
    # Block 4: A[:, 2:4], B[2:4, 0:2], C[:, 0:2]
    # Block 5: A[:, 2:4], B[2:4, 2:4], C[:, 2:4]
    # Block 6: A[:, 2:4], B[2:4, 4:6], C[:, 4:6]
    # Block 7: A[:, 2:4], B[2:4, 6:8], C[:, 6:8]
    
    blocks = [
        (0, 0, 0), # a_start, b_row_start, b_col_start
        (0, 0, 2),
        (0, 0, 4),
        (0, 0, 6),
        (2, 2, 0),
        (2, 2, 2),
        (2, 2, 4),
        (2, 2, 6),
    ]
    
    for a_start, b_row_start, b_col_start in blocks:
        a_cols = [a_start, a_start + 1]
        b_rows = [b_row_start, b_row_start + 1]
        b_cols = [b_col_start, b_col_start + 1]
        c_cols = [b_col_start, b_col_start + 1]
        
        # Strassen multipliers
        multipliers = [
            # M1: (x00 + x11) * (y00 + y11)
            ([(0, a_cols[0]), (1, a_cols[1])], 
             [(b_rows[0], b_cols[0]), (b_rows[1], b_cols[1])],
             [(c_cols[0], 0, 1), (c_cols[1], 1, 1)]),
             
            # M2: (x10 + x11) * y00
            ([(1, a_cols[0]), (1, a_cols[1])], 
             [(b_rows[0], b_cols[0])],
             [(c_cols[0], 1, 1)]),
             
            # M3: x00 * (y01 - y11)
            ([(0, a_cols[0])], 
             [(b_rows[0], b_cols[1]), (b_rows[1], b_cols[1])],
             [(c_cols[1], 0, 1), (c_cols[1], 1, 1)]),
             
            # M4: x11 * (y10 - y00)
            ([(1, a_cols[1])], 
             [(b_rows[1], b_cols[0]), (b_rows[0], b_cols[0])],
             [(c_cols[0], 0, 1), (c_cols[0], 1, 1)]),
             
            # M5: (x00 + x01) * y11
            ([(0, a_cols[0]), (0, a_cols[1])], 
             [(b_rows[1], b_cols[1])],
             [(c_cols[0], 0, -1), (c_cols[1], 0, 1)]),
             
            # M6: (x10 - x00) * (y00 + y01)
            ([(1, a_cols[0]), (0, a_cols[0])], 
             [(b_rows[0], b_cols[0]), (b_rows[0], b_cols[1])],
             [(c_cols[1], 1, 1)]),
             
            # M7: (x01 - x11) * (y10 + y11)
            ([(0, a_cols[1]), (1, a_cols[1])], 
             [(b_rows[1], b_cols[0]), (b_rows[1], b_cols[1])],
             [(c_cols[0], 0, 1)]),
        ]
        
        for u, v, w in multipliers:
            for i, j in u:
                U[i * m + j][col] = 1
            for j, k in v:
                V[j * p + k][col] = 1
            for k, i, coeff in w:
                W[k * n + i][col] = coeff
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
