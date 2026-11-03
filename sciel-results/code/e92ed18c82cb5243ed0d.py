"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The problem is 2x4x4 matrix multiplication.
    # We decompose it into 4 independent 2x2x2 Strassen multiplications.
    # Rank = 4 * 7 = 28.
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2:
    # U_coeffs, V_coeffs, W_coeffs
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Block definitions:
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # Offsets for each block:
    # A: (row_offset, col_offset)
    # B: (row_offset, col_offset)
    # C: (row_offset, col_offset)
    # Note: A is 2x4, B is 4x4, C is 2x4.
    # A row offset is always 0 (since A has 2 rows).
    # B row offset is 0 or 2.
    # C col offset is 0 or 2.
    
    blocks = [
        # (A_col_off, B_row_off, B_col_off, C_col_off)
        (0, 0, 0, 0), # Block 0
        (2, 2, 0, 0), # Block 1
        (0, 0, 2, 2), # Block 2
        (2, 2, 2, 2), # Block 3
    ]
    
    for b_idx, (a_col_off, b_row_off, b_col_off, c_col_off) in enumerate(blocks):
        for s_idx, (u_coeffs, v_coeffs, w_coeffs) in enumerate(strassen_terms):
            col = b_idx * 7 + s_idx
            
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                # Global A index: i * 4 + j
                i = li
                j = a_col_off + lj
                g_idx = i * 4 + j
                U[g_idx][col] += coeff
                
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                # Global B index: j * 4 + r
                j = b_row_off + li
                r = b_col_off + lj
                g_idx = j * 4 + r
                V[g_idx][col] += coeff
                
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                # Global C index: r * 2 + i
                i = li
                r = c_col_off + lj
                g_idx = r * 2 + i
                W[g_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
