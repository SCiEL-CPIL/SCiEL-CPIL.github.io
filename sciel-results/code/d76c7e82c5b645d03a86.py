def add_strassen_2x2x2_flex(U, V, W, a_blocks, b_blocks, c_blocks, col_start):
    # a_blocks: list of (br, bc, coeff)
    # b_blocks: list of (br, bc, coeff)
    # c_blocks: list of (br, bc, coeff)
    u2, v2, w2 = get_strassen_2x2x2()
    
    for c in range(7):
        for r in range(4):
            # U
            if u2[r][c] != 0:
                a_row_local = r // 2
                a_col_local = r % 2
                for (br, bc, coeff) in a_blocks:
                    a_row = br * 2 + a_row_local
                    a_col = bc * 2 + a_col_local
                    u_idx = a_row * 5 + a_col
                    U[u_idx][col_start + c] += coeff * u2[r][c]
            
            # V
            if v2[r][c] != 0:
                b_row_local = r // 2
                b_col_local = r % 2
                for (br, bc, coeff) in b_blocks:
                    b_row = br * 2 + b_row_local
                    b_col = bc * 2 + b_col_local
                    v_idx = b_row * 5 + b_col
                    V[v_idx][col_start + c] += coeff * v2[r][c]
            
            # W
            if w2[r][c] != 0:
                c_row_local = r // 2
                c_col_local = r % 2
                for (br, bc, coeff) in c_blocks:
                    c_row = br * 2 + c_row_local
                    c_col = bc * 2 + c_col_local
                    w_idx = c_row * 4 + c_col
                    W[w_idx][col_start + c] += coeff * w2[r][c]
