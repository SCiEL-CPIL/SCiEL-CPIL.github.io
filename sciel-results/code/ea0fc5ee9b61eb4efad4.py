"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B = [[c00, c01], [c10, c11]]
    #
    # M0 = (a00+a11)*(b00+b11)
    # M1 = (a10+a11)*b00
    # M2 = a00*(b01-b11)
    # M3 = a11*(b10-b00)
    # M4 = (a00+a01)*b11
    # M5 = (a10-a00)*(b00+b01)
    # M6 = (a01-a11)*(b10+b11)
    #
    # c00 = M0 + M3 - M4 + M6
    # c01 = M2 + M4
    # c10 = M1 + M3
    # c11 = M0 - M1 + M2 + M5
    
    # U for 2x2: rows are [a00, a01, a10, a11], cols are 7 Strassen terms
    U2 = [
        [1,  0,  1,  0,  1, -1,  0],  # a00
        [0,  0,  0,  0,  1,  0,  1],  # a01
        [0,  1,  0,  0,  0,  1,  0],  # a10
        [1,  1,  0,  1,  0,  0, -1],  # a11
    ]
    # V for 2x2: rows are [b00, b01, b10, b11], cols are 7 Strassen terms
    V2 = [
        [1,  1,  0, -1,  0,  1,  0],  # b00
        [0,  0,  1,  0,  0,  1,  0],  # b01
        [0,  0,  0,  1,  0,  0,  1],  # b10
        [1,  0, -1,  0,  1,  0,  1],  # b11
    ]
    # W for 2x2: rows are [c00, c01, c10, c11], cols are 7 Strassen terms
    W2 = [
        [1,  0,  0,  1, -1,  0,  1],  # c00
        [0,  0,  1,  0,  1,  0,  0],  # c01
        [0,  1,  0,  1,  0,  0,  0],  # c10
        [1, -1, 1,  0,  0,  1,  0],  # c11
    ]
    
    # Build 6x6 from 3x3 blocks of 2x2
    rank = 27 * 7  # 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for i_b in range(3):
        for j_b in range(3):
            for k_b in range(3):
                for s in range(7):
                    col = (i_b*3 + j_b)*7 + s
                    for i_l in range(2):
                        for j_l in range(2):
                            a_row = i_b*2 + i_l
                            a_col = j_b*2 + j_l
                            u_idx = a_row*6 + a_col
                            U[u_idx][col] = U2[i_l*2 + j_l][s]
                    for j_l in range(2):
                        for k_l in range(2):
                            b_row = j_b*2 + j_l
                            b_col = k_b*2 + k_l
                            v_idx = b_row*6 + b_col
                            V[v_idx][col] = V2[j_l*2 + k_l][s]
                    for i_l in range(2):
                        for k_l in range(2):
                            c_row = i_b*2 + i_l
                            c_col = k_b*2 + k_l
                            w_idx = c_col*6 + c_row  # W[k*n+i, r]
                            W[w_idx][col] = W2[i_l*2 + k_l][s]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
