"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * 7 = 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get index in U for A[i, j]
    def u_idx(i, j):
        return i * m + j
        
    # Helper to get index in V for B[i, j]
    def v_idx(i, j):
        return i * p + j
        
    # Helper to get index in W for C[i, j] (row i, col j)
    # Convention: W[k*n + i, r] where i is row of A, k is col of B.
    # So index is col * n + row.
    def w_idx(i, j):
        return j * n + i

    # Define the 7 outer Strassen terms
    # Each term is defined by (A_comb, B_comb, output_comb)
    # A_comb is a list of (block, coeff)
    # Blocks are identified by (block_row, block_col) where block_row, block_col in {0, 1}
    # 0: top/left, 1: bottom/right
    
    # Outer Strassen definitions:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    outer_terms = [
        # (A_blocks, B_blocks, Output_blocks)
        # A_blocks: list of (block_id, coeff)
        # B_blocks: list of (block_id, coeff)
        # Output_blocks: list of (block_id, coeff)
        # Block IDs: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        # M1
        {
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, -1), (1, 1), (2, 1)] # C11 += M1, C22 += -M1? No.
            # Let's re-verify output formulas.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
        },
        # M2
        {
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1), (3, -1)] # C21 += M2, C22 += -M2
        },
        # M3
        {
            'A': [(0, 1)],
            'B': [(1, 1), (3, -1)],
            'C': [(1, 1), (3, 1)] # C12 += M3, C22 += M3
        },
        # M4
        {
            'A': [(3, 1)],
            'B': [(2, 1), (0, -1)],
            'C': [(0, 1), (2, 1)] # C11 += M4, C21 += M4
        },
        # M5
        {
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(0, -1), (1, 1)] # C11 += -M5, C12 += M5
        },
        # M6
        {
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(3, 1)] # C22 += M6
        },
        # M7
        {
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)],
            'C': [(0, 1)] # C11 += M7
        }
    ]
    
    # Correcting Output Coefficients based on standard Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's hardcode the contributions to C blocks for each M_t
    # M_t contributes to C_block with coeff c
    # M1: C00: 1, C11: 1
    # M2: C10: 1, C11: -1
    # M3: C01: 1, C11: 1
    # M4: C00: 1, C10: 1
    # M5: C00: -1, C01: 1
    # M6: C11: 1
    # M7: C00: 1
    
    c_contribs = [
        [(0, 1), (3, 1)],      # M1
        [(2, 1), (3, -1)],     # M2
        [(1, 1), (3, 1)],      # M3
        [(0, 1), (2, 1)],      # M4
        [(0, -1), (1, 1)],     # M5
        [(3, 1)],              # M6
        [(0, 1)]               # M7
    ]

    # Base case Strassen products for 2x2 matrices
    # P1 = (x11 + x22)(y11 + y22)
    # P2 = (x21 + x22)y11
    # P3 = x11(y12 - y22)
    # P4 = x22(y21 - y11)
    # P5 = (x11 + x12)y22
    # P6 = (x21 - x11)(y11 + y12)
    # P7 = (x12 - x22)(y21 + y22)
    
    # For each P_s, we define which elements of X (2x2) and Y (2x2) are involved.
    # X elements: (0,0), (0,1), (1,0), (1,1)
    # Y elements: (0,0), (0,1), (1,0), (1,1)
    
    inner_products = [
        # (X_coeffs, Y_coeffs)
        # X_coeffs: dict {(r,c): coeff}
        # Y_coeffs: dict {(r,c): coeff}
        
        # P1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # P2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # P3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # P4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        # P5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # P6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # P7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    ]

    # Map block ID to element offsets in the 4x4 matrix
    # Block 0: (0,0) -> rows 0-1, cols 0-1
    # Block 1: (0,1) -> rows 0-1, cols 2-3
    # Block 2: (1,0) -> rows 2-3, cols 0-1
    # Block 3: (1,1) -> rows 2-3, cols 2-3
    
    def get_block_elements(block_id):
        # Returns list of (i, j) for the 4 elements in the block
        br, bc = divmod(block_id, 2)
        base_r = br * 2
        base_c = bc * 2
        return [
            (base_r + 0, base_c + 0),
            (base_r + 0, base_c + 1),
            (base_r + 1, base_c + 0),
            (base_r + 1, base_c + 1)
        ]
    
    # Iterate through outer terms
    for t_idx, term in enumerate(outer_terms):
        A_blocks = term['A']
        B_blocks = term['B']
        C_blocks = c_contribs[t_idx]
        
        # Iterate through inner products
        for s_idx, (X_coeffs, Y_coeffs) in enumerate(inner_products):
            rank_idx = t_idx * 7 + s_idx
            
            # Accumulate U and V contributions
            # U corresponds to A
            # V corresponds to B
            
            # For each element in the 2x2 X matrix (which is a combination of blocks)
            # We need to map X element (r,c) to A elements.
            # X is a 2x2 matrix formed by linear combinations of A blocks.
            # X[r,c] = sum_{b} coeff_b * A_block_b[r,c]
            
            # Let's build a dictionary for U contributions: { (i,j): coeff }
            U_contrib = {}
            V_contrib = {}
            
            # Process X (A side)
            for (xr, xc), x_coeff in X_coeffs.items():
                # This X element is a sum of block elements
                # Wait, X itself is a matrix.
                # In Strassen, X is a 2x2 matrix.
                # The elements of X are linear combinations of elements of A.
                # Specifically, if X = A_block_1 + A_block_2, then X[r,c] = A_block_1[r,c] + A_block_2[r,c].
                
                # We need to sum contributions from all A_blocks involved in this outer term.
                # But wait, the outer term defines X as a sum of blocks.
                # e.g. M1: X = A00 + A11.
                # So X[r,c] = A00[r,c] + A11[r,c].
                
                # We iterate over the blocks in A_blocks
                for b_id, b_coeff in A_blocks:
                    # The element (xr, xc) of the block b_id is at global position:
                    elems = get_block_elements(b_id)
                    # The block element corresponding to local (xr, xc)
                    # Local (0,0) -> global (base_r, base_c)
                    # Local (0,1) -> global (base_r, base_c+1)
                    # Local (1,0) -> global (base_r+1, base_c)
                    # Local (1,1) -> global (base_r+1, base_c+1)
                    # So index in elems list is xr * 2 + xc
                    
                    global_i, global_j = elems[xr * 2 + xc]
                    total_coeff = x_coeff * b_coeff
                    
                    # Add to U_contrib
                    key = (global_i, global_j)
                    if key in U_contrib:
                        U_contrib[key] += total_coeff
                    else:
                        U_contrib[key] = total_coeff

            # Process Y (B side)
            for (yr, yc), y_coeff in Y_coeffs.items():
                for b_id, b_coeff in B_blocks:
                    elems = get_block_elements(b_id)
                    global_i, global_j = elems[yr * 2 + yc]
                    total_coeff = y_coeff * b_coeff
                    
                    key = (global_i, global_j)
                    if key in V_contrib:
                        V_contrib[key] += total_coeff
                    else:
                        V_contrib[key] = total_coeff

            # Fill U and V
            for (i, j), coeff in U_contrib.items():
                if coeff != 0:
                    U[u_idx(i, j)][rank_idx] = coeff
                    
            for (i, j), coeff in V_contrib.items():
                if coeff != 0:
                    V[v_idx(i, j)][rank_idx] = coeff

            # Fill W
            # The result of this inner product P_s contributes to the output blocks C.
            # The output of the base case Strassen P_s is a 2x2 matrix Z.
            # We need to know which element of Z (local r,c) P_s contributes to.
            # And then map that to the global C blocks.
            
            # Base case outputs:
            # P1 contributes to Z00 (+1), Z11 (+1)
            # P2 contributes to Z10 (+1), Z11 (-1)
            # P3 contributes to Z01 (+1), Z11 (+1)
            # P4 contributes to Z00 (+1), Z10 (+1)
            # P5 contributes to Z00 (-1), Z01 (+1)
            # P6 contributes to Z11 (+1)
            # P7 contributes to Z00 (+1)
            
            z_contribs = [
                [(0,0, 1), (1,1, 1)],
                [(1,0, 1), (1,1, -1)],
                [(0,1, 1), (1,1, 1)],
                [(0,0, 1), (1,0, 1)],
                [(0,0, -1), (0,1, 1)],
                [(1,1, 1)],
                [(0,0, 1)]
            ]
            
            z_parts = z_contribs[s_idx]
            
            for (zr, zc, z_coeff) in z_parts:
                # This P_s contributes z_coeff to the element (zr, zc) of the 2x2 result block.
                # The 2x2 result block is a linear combination of global C blocks.
                # The outer term t defines which global C blocks are affected.
                # C_blocks is a list of (block_id, coeff).
                # The local element (zr, zc) of the result matrix corresponds to the same local position in the global C blocks.
                
                for c_id, c_coeff in C_blocks:
                    elems = get_block_elements(c_id)
                    global_i, global_j = elems[zr * 2 + zc]
                    total_coeff = z_coeff * c_coeff
                    
                    # W index is j * n + i
                    w_i = w_idx(global_i, global_j)
                    W[w_i][rank_idx] += total_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
