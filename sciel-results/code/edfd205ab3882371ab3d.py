"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part, Rank 16 for 4x4x1 part. Total 65.
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition matrices (4x7)
    # U: rows 0-3 (A00, A01, A10, A11)
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    # V: rows 0-3 (B00, B01, B10, B11)
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    # W: rows 0-3 (C00, C01, C10, C11)
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Define sub-matrix index mappings for 4x4 -> 2x2 blocks
    # Block (br, bc) where br, bc in {0, 1}
    # Global indices for A (4x4): row r, col c -> r*4+c
    # Local indices for 2x2: lr, lc -> lr*2+lc
    # A11 (br=0, bc=0): Global rows 0-1, cols 0-1
    # A12 (br=0, bc=1): Global rows 0-1, cols 2-3
    # A21 (br=1, bc=0): Global rows 2-3, cols 0-1
    # A22 (br=1, bc=1): Global rows 2-3, cols 2-3
    
    def get_global_indices(block_r, block_c):
        # Returns list of 4 global indices corresponding to local 0,1,2,3
        base_r = block_r * 2
        base_c = block_c * 2
        return [
            base_r * 4 + base_c,       # 0,0
            base_r * 4 + base_c + 1,   # 0,1
            (base_r + 1) * 4 + base_c, # 1,0
            (base_r + 1) * 4 + base_c + 1 # 1,1
        ]
    
    # Strassen steps for 4x4x4
    # Each step: (A_blocks, B_blocks, C_contributions)
    # A_blocks: dict {block_idx: coeff}
    # B_blocks: dict {block_idx: coeff}
    # C_contributions: dict {block_idx: coeff}
    
    # Block indices: 0=A11, 1=A12, 2=A21, 3=A22
    steps = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}), # Wait, C21 += M2, C22 -= M2
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}), # C12 += M3, C22 += M3
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}), # C11 += M4, C21 += M4
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}), # C11 -= M5, C12 += M5
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # C22 += M6
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})  # C11 += M7
    ]
    
    # Map block index to (br, bc)
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    block_to_coords = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1)
    }
    
    for s_idx, (A_coeffs, B_coeffs, C_coeffs) in enumerate(steps):
        # Get the linear combination of blocks for A and B
        # We need to form the local 2x2 matrix A_in and B_in
        # A_in[local_r][local_c] = sum over blocks of coeff * A_block[local_r][local_c]
        # Since all blocks are 2x2, we can just sum the coefficients for each local position?
        # No, the blocks are disjoint in the 4x4 matrix.
        # So A_in is a 2x2 matrix where each element is a linear combo of elements from different global blocks.
        # But the Strassen decomposition treats A_in as a single 2x2 matrix.
        # The U matrix for the local decomposition picks out linear combos of the 4 elements of A_in.
        # So we need to map the local U coefficients to global U indices.
        
        # For each local term t in 0..6:
        for t in range(7):
            col = s_idx * 7 + t
            
            # Accumulate U
            # U_global[row, col] += sum over blocks b of (A_coeffs[b] * U_strassen[local_row, t] * mapping)
            # Actually, U_strassen[local_row, t] is the coefficient for the local element at local_row.
            # The local element at local_row corresponds to a specific element in the 2x2 block.
            # But A_in is formed by summing blocks.
            # A_in[lr][lc] = sum_b A_coeffs[b] * A_b[lr][lc]
            # The local Strassen decomposition uses U_strassen which is linear in the 4 elements of A_in.
            # U_in[t] = sum_{lr} U_strassen[lr, t] * A_in[flat_lr]
            # = sum_{lr} U_strassen[lr, t] * (sum_b A_coeffs[b] * A_b[flat_lr])
            # = sum_b A_coeffs[b] * (sum_{lr} U_strassen[lr, t] * A_b[flat_lr])
            # The term in parentheses is the contribution of block b to the t-th linear form.
            # For a specific block b, the global indices for its 4 elements are known.
            # Let global_indices_b = [g0, g1, g2, g3] corresponding to local 0,1,2,3.
            # Then for block b, the contribution to U_global[g_k, col] is A_coeffs[b] * U_strassen[k, t].
            
            for b_idx, coeff_a in A_coeffs.items():
                br, bc = block_to_coords[b_idx]
                g_indices = get_global_indices(br, bc)
                for k in range(4):
                    U[g_indices[k]][col] += coeff_a * U_strassen[k][t]
            
            # Accumulate V
            for b_idx, coeff_b in B_coeffs.items():
                br, bc = block_to_coords[b_idx]
                # B is 4x5. The first 4 cols are used.
                # Block indices for B are the same as A for the first 4 cols.
                g_indices = get_global_indices(br, bc)
                # Note: B indices are j*5+k. Here k is the col in 4x4 part (0-3).
                # So g_indices are correct for B's first 4 cols.
                for k in range(4):
                    V[g_indices[k]][col] += coeff_b * V_strassen[k][t]
            
            # Accumulate W
            # W_global[row, col] += sum over blocks b of (C_coeffs[b] * W_strassen[local_row, t] * mapping)
            for b_idx, coeff_c in C_coeffs.items():
                br, bc = block_to_coords[b_idx]
                g_indices = get_global_indices(br, bc)
                for k in range(4):
                    W[g_indices[k]][col] += coeff_c * W_strassen[k][t]
    
    # Add 4x4x1 part (rank 16)
    # Cols 49..64
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            # A index: i*4 + j
            U[i * 4 + j][col] = 1
            # B index: j*5 + 4 (col 4)
            V[j * 5 + 4][col] = 1
            # W index: 4*4 + i (row 4, col i)
            W[16 + i][col] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
