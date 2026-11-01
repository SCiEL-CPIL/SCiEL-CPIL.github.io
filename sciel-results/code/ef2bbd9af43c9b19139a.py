"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    block_size = 2
    
    # Strassen decomposition for 2x2 matrix multiplication
    # Terms: (u_coeffs, v_coeffs, w_coeffs)
    # u: [a11, a12, a21, a22]
    # v: [b11, b12, b21, b22]
    # w: [c11, c12, c21, c22]
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # P3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # P5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # P7
    ]
    
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    mapping_v = [
        (0, 0), # b11
        (0, 1), # b12
        (1, 0), # b21
        (1, 1)  # b22
    ]
    
    mapping_w = [
        (0, 0), # c11
        (0, 1), # c12
        (1, 0), # c21
        (1, 1)  # c22
    ]
    
    col = 0
    for block in range(num_blocks):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            
            for idx, val in enumerate(u_coeffs):
                if val != 0:
                    U[idx][col] = val
            
            for idx, val in enumerate(v_coeffs):
                if val != 0:
                    j_loc, k_loc = mapping_v[idx]
                    row_v = j_loc * p + (block * 2 + k_loc)
                    V[row_v][col] = val
            
            for idx, val in enumerate(w_coeffs):
                if val != 0:
                    i_loc, k_loc = mapping_w[idx]
                    row_w = (block * 2 + k_loc) * n + i_loc
                    W[row_w][col] = val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
