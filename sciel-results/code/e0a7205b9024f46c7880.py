"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 24 decomposition using 4 Strassen 2x2x2 blocks
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1..M7
    # U coeffs (A), V coeffs (B), W coeffs (C)
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: dict {(local_i, local_j): coeff}
        # v_coeffs: dict {(local_j, local_k): coeff}
        # w_coeffs: dict {(local_i, local_k): coeff}
        # Note: w_coeffs are for the final C, summed over terms.
        # We store the contribution of each term to C.
        
        # Term 1
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # Term 2
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        # Term 3
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        # Term 4
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        # Term 5
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        # Term 6
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # Term 7
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        )
    ]
    
    # Blocks: (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
    # A is 2x4. B is 4x4. C is 2x4.
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2)
    ]
    
    col = 0
    for b_idx in range(4):
        a_ro, a_co, b_ro, b_co, c_ro, c_co = blocks[b_idx]
        
        for t_idx in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t_idx]
            
            # Fill U
            for (li, lj), coeff in u_coeffs.items():
                gi = a_ro + li
                gj = a_co + lj
                U[gi * m + gj][col] += coeff
            
            # Fill V
            for (lj, lk), coeff in v_coeffs.items():
                gj = b_ro + lj
                gk = b_co + lk
                V[gj * p + gk][col] += coeff
            
            # Fill W
            for (li, lk), coeff in w_coeffs.items():
                gi = c_ro + li
                gk = c_co + lk
                # W index: gk * n + gi
                W[gk * n + gi][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
