"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the problem into two independent 2x4x2 multiplications
    # C_L = A * B[:, 0:2]
    # C_R = A * B[:, 2:4]
    # Each 2x4x2 multiplication is decomposed into two 2x2x2 Strassen multiplications
    # Total rank = 2 * 2 * 7 = 28
    
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # U: dict {local_idx: coeff}, V: dict {local_idx: coeff}, W: dict {local_idx: coeff}
    # local_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # Block 0: C_L contribution from A_L * B_LL
    # A_L: rows 0,1 of A, cols 0,1 of A. (Actually A is 2x4, so A_L is cols 0,1)
    # B_LL: rows 0,1 of B, cols 0,1 of B.
    # C_L: cols 0,1 of C.
    
    # Block 1: C_L contribution from A_R * B_RL
    # A_R: cols 2,3 of A.
    # B_RL: rows 2,3 of B, cols 0,1 of B.
    
    # Block 2: C_R contribution from A_L * B_LR
    # A_L: cols 0,1 of A.
    # B_LR: rows 0,1 of B, cols 2,3 of B.
    
    # Block 3: C_R contribution from A_R * B_RR
    # A_R: cols 2,3 of A.
    # B_RR: rows 2,3 of B, cols 2,3 of B.

    def add_block(a_col_offset, b_row_offset, b_col_offset, c_col_offset, base_rank):
        for s in range(7):
            col = base_rank + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]
            
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                # A global index: i * 4 + j
                # i = li (0 or 1)
                # j = a_col_offset + lj
                g_idx_a = li * 4 + (a_col_offset + lj)
                U[g_idx_a][col] += coeff
                
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                # B global index: j * 4 + r
                # j = b_row_offset + li
                # r = b_col_offset + lj
                g_idx_b = (b_row_offset + li) * 4 + (b_col_offset + lj)
                V[g_idx_b][col] += coeff
                
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                # C global index: r * 2 + i
                # r = c_col_offset + lj
                # i = li
                g_idx_c = (c_col_offset + lj) * 2 + li
                W[g_idx_c][col] += coeff

    # Block 0: A_L (cols 0,1), B_LL (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    add_block(0, 0, 0, 0, 0)
    
    # Block 1: A_R (cols 2,3), B_RL (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    add_block(2, 2, 0, 0, 7)
    
    # Block 2: A_L (cols 0,1), B_LR (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    add_block(0, 0, 2, 2, 14)
    
    # Block 3: A_R (cols 2,3), B_RR (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    add_block(2, 2, 2, 2, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
