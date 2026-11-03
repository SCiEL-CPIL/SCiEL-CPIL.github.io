"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # u, v, w are length 4 vectors (flattened 2x2)
    # Flattening order: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])   # M7
    ]
    
    # Define block mappings for 4x4 matrices
    # Block (i, j) of 4x4 matrix corresponds to indices in flattened 16-vector
    # Row-major: index = r * 4 + c
    def get_block_indices(block_i, block_j):
        start_r = block_i * 2
        start_c = block_j * 2
        indices = []
        for dr in range(2):
            for dc in range(2):
                r = start_r + dr
                c = start_c + dc
                indices.append(r * 4 + c)
        return indices

    block_indices = {}
    for bi in range(2):
        for bj in range(2):
            block_indices[(bi, bj)] = get_block_indices(bi, bj)

    # Strassen formulas for 4x4x4 in terms of 2x2 blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Each M_i is a product of two linear combinations of 2x2 blocks.
    # We represent the linear combination as a list of (coeff, block_idx) pairs.
    # For U (A side):
    # M1_U: A00 + A11 -> [(1, (0,0)), (1, (1,1))]
    # M2_U: A10 + A11 -> [(1, (1,0)), (1, (1,1))]
    # M3_U: A00 -> [(1, (0,0))]
    # M4_U: A11 -> [(1, (1,1))]
    # M5_U: A00 + A01 -> [(1, (0,0)), (1, (0,1))]
    # M6_U: A10 - A00 -> [(1, (1,0)), (-1, (0,0))]
    # M7_U: A01 - A11 -> [(1, (0,1)), (-1, (1,1))]
    
    # For V (B side):
    # M1_V: B00 + B11 -> [(1, (0,0)), (1, (1,1))]
    # M2_V: B00 -> [(1, (0,0))]
    # M3_V: B01 - B11 -> [(1, (0,1)), (-1, (1,1))]
    # M4_V: B10 - B00 -> [(1, (1,0)), (-1, (0,0))]
    # M5_V: B11 -> [(1, (1,1))]
    # M6_V: B00 + B01 -> [(1, (0,0)), (1, (0,1))]
    # M7_V: B10 + B11 -> [(1, (1,0)), (1, (1,1))]
    
    # For W (C side):
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # W coefficients for each block of C
    # C00 (block 0,0): M1:1, M4:-1, M5:-1, M7:1
    # C01 (block 0,1): M3:1, M5:1
    # C10 (block 1,0): M2:1, M4:1
    # C11 (block 1,1): M1:1, M2:-1, M3:1, M6:-1
    
    w_coeffs = {
        (0, 0): {0: 1, 3: -1, 4: -1, 6: 1}, # M1, M4, M5, M7 (indices 0,3,4,6 in 0-6)
        (0, 1): {2: 1, 4: 1},               # M3, M5
        (1, 0): {1: 1, 3: 1},               # M2, M4
        (1, 1): {0: 1, 1: -1, 2: 1, 5: -1}  # M1, M2, M3, M6
    }
    
    u_coeffs = [
        [(1, (0,0)), (1, (1,1))], # M1
        [(1, (1,0)), (1, (1,1))], # M2
        [(1, (0,0))],             # M3
        [(1, (1,1))],             # M4
        [(1, (0,0)), (1, (0,1))], # M5
        [(1, (1,0)), (-1, (0,0))],# M6
        [(1, (0,1)), (-1, (1,1))] # M7
    ]
    
    v_coeffs = [
        [(1, (0,0)), (1, (1,1))], # M1
        [(1, (0,0))],             # M2
        [(1, (0,1)), (-1, (1,1))],# M3
        [(1, (1,0)), (-1, (0,0))],# M4
        [(1, (1,1))],             # M5
        [(1, (0,0)), (1, (0,1))], # M6
        [(1, (1,0)), (1, (1,1))]  # M7
    ]
    
    rank = 98 # 2 * 49
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0.0] * rank for _ in range(p * n)]  # 32 x 98
    
    for block in range(2): # Left and Right halves of B
        base_col = block * 49
        
        for m_idx in range(7): # 7 Strassen products for 4x4
            for s_idx in range(7): # 7 Strassen products for 2x2
                col = base_col + m_idx * 7 + s_idx
                
                # Get the 2x2 Strassen vectors for this inner product
                u_2, v_2, w_2 = strassen_2x2[s_idx]
                
                # Accumulate U contributions
                # u_2 is for the 2x2 matrix X_m
                # X_m is a linear combination of A blocks
                for coeff, a_block in u_coeffs[m_idx]:
                    u_indices = block_indices[a_block]
                    for k, idx in enumerate(u_indices):
                        U[idx][col] += coeff * u_2[k]
                        
                # Accumulate V contributions
                for coeff, b_block in v_coeffs[m_idx]:
                    v_indices = block_indices[b_block]
                    for k, idx in enumerate(v_indices):
                        b_row = idx // 4
                        b_col_local = idx % 4
                        b_col_global = block * 4 + b_col_local
                        v_flat_idx = b_row * p + b_col_global
                        V[v_flat_idx][col] += coeff * v_2[k]
                        
                # Accumulate W contributions
                # w_2 is for the 2x2 matrix Y_m (part of M_m)
                # M_m contributes to C blocks with certain coefficients
                # We need to add w_2 scaled by the W coefficient for M_m to each C block
                # But w_2 is a 2x2 matrix itself, corresponding to the output of the inner Strassen step.
                # The inner Strassen step computes the 2x2 product.
                # So w_2[k] is the coefficient for the k-th element of the 2x2 output block.
                
                # The output of M_m is a 2x2 matrix.
                # This 2x2 matrix is added to specific 2x2 blocks of C.
                # Let's iterate over the C blocks that M_m affects.
                
                # Wait, the w_2 vector from strassen_2x2 corresponds to the 2x2 output of the *inner* multiplication.
                # The inner multiplication is $X_m Y_m$.
                # The result is a 2x2 matrix.
                # This result is $M_m$.
                # $M_m$ is then combined to form the 4x4 result C.
                
                # So, for each C block (c_i, c_j) that M_m affects:
                # We add $w_{coeff} \times w_2$ to the corresponding positions in W.
                
                # Let's find which C blocks are affected by M_m.
                # We can just iterate all 4 C blocks and check if M_m has a coefficient.
                for c_bi in range(2):
                    for c_bj in range(2):
                        if m_idx in w_coeffs[(c_bi, c_bj)]:
                            w_coeff = w_coeffs[(c_bi, c_bj)][m_idx]
                            c_indices = block_indices[(c_bi, c_bj)]
                            for k, idx in enumerate(c_indices):
                                c_row = idx // 4
                                c_col_local = idx % 4
                                c_col_global = block * 4 + c_col_local
                                w_flat_idx = c_col_global * n + c_row
                                W[w_flat_idx][col] += w_coeff * w_2[k]
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
