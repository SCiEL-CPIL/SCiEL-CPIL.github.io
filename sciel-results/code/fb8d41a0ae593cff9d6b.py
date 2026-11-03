import numpy as np

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2 factors
    # u, v, w are lists of 7 vectors of length 4
    s_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    s_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    s_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Helper to add a Strassen block
    # a_rows: start row of A block
    # a_cols: start col of A block
    # b_rows: start row of B block
    # b_cols: start col of B block
    # c_rows: start row of C block
    # c_cols: start col of C block
    # col_offset: current column in U,V,W
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c, col_off):
        for t in range(7):
            col = col_off + t
            # Fill U
            for lr in range(2):
                for lc in range(2):
                    val = s_u[t][lr*2+lc]
                    if val != 0:
                        U[(a_r + lr) * m + (a_c + lc), col] += val
            # Fill V
            for lr in range(2):
                for lc in range(2):
                    val = s_v[t][lr*2+lc]
                    if val != 0:
                        V[(b_r + lr) * p + (b_c + lc), col] += val
            # Fill W
            for lr in range(2):
                for lc in range(2):
                    val = s_w[t][lr*2+lc]
                    if val != 0:
                        # Output index is c_col * n + c_row
                        # c_r is row, c_c is col
                        W[(c_c + lc) * n + (c_r + lr), col] += val
                        
    # Part 1: 2x4x4 using 4 Strassen blocks
    # A_top is rows 0,1 of A.
    # A_11: rows 0,1 cols 0,1
    # A_12: rows 0,1 cols 2,3
    # B_11: rows 0,1 cols 0,1
    # B_12: rows 0,1 cols 2,3
    # B_21: rows 2,3 cols 0,1
    # B_22: rows 2,3 cols 2,3
    # C_top is rows 0,1 of C.
    # P1 (A11*B11) -> C rows 0,1 cols 0,1
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    # P2 (A11*B12) -> C rows 0,1 cols 2,3
    add_strassen_block(0, 0, 0, 2, 0, 2, 7)
    # P3 (A12*B21) -> C rows 0,1 cols 0,1
    add_strassen_block(0, 2, 2, 0, 0, 0, 14)
    # P4 (A12*B22) -> C rows 0,1 cols 2,3
    add_strassen_block(0, 2, 2, 2, 0, 2, 21)
    
    # Part 2: 1x4x4 schoolbook
    # A_bot is row 2 of A.
    # C_bot is row 2 of C.
    col_off = 28
    for j in range(4):
        for k in range(4):
            col = col_off + j * 4 + k
            # U: index 2*m + j = 8 + j
            U[2 * m + j, col] = 1
            # V: index j*p + k
            V[j * p + k, col] = 1
            # W: index k*n + 2
            W[k * n + 2, col] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
