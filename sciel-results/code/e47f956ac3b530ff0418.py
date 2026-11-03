import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4, Rank 16 for 4x4x1
    rank = 49 + 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 term
    # x_coeffs: dict {(r,c): coeff} for the 2x2 matrix X
    # y_coeffs: dict {(r,c): coeff} for the 2x2 matrix Y
    # w_coeffs: dict {(r,c): coeff} for the 2x2 matrix Z (output)
    # base_x: (i_off, j_off) offset in A
    # base_y: (j_off, k_off) offset in B
    # base_z: (k_off, i_off) offset in C
    def add_strassen_2x2(x_coeffs, y_coeffs, w_coeffs, base_x, base_y, base_z):
        nonlocal col
        # The 7 Strassen terms
        # M1 = (x11+x22)(y11+y22)
        # M2 = (x21+x22)y11
        # M3 = x11(y12-y22)
        # M4 = x22(y21-y11)
        # M5 = (x11+x12)y22
        # M6 = (x21-x11)(y11+y12)
        # M7 = (x12-x22)(y21+y22)
        
        terms = [
            # (x_part, y_part, w_part)
            # x_part is a list of (coeff, x_idx)
            # w_part is a list of (coeff, z_idx)
            # We need to combine these into a single rank-1 term?
            # No, each M_k is a product of a linear form in X and a linear form in Y.
            # So it is a single rank-1 term in the 2x2x2 tensor.
            # The contribution to Z is w_coeffs * M_k.
            
            # So for each k, we define:
            # U_vec: non-zero at A indices corresponding to X linear form.
            # V_vec: non-zero at B indices corresponding to Y linear form.
            # W_vec: non-zero at C indices corresponding to Z linear form (scaled by w_coeffs).
            
            # Let's define the linear forms for X and Y for each M_k.
            # X indices: (0,0), (0,1), (1,0), (1,1)
            # Y indices: (0,0), (0,1), (1,0), (1,1)
            # Z indices: (0,0), (0,1), (1,0), (1,1)
            
            # M1
            {
                'x': {0: 1, 3: 1}, # x11, x22
                'y': {0: 1, 3: 1}, # y11, y22
                'w': {0: 1, 3: 1}  # z11, z22
            },
            # M2
            {
                'x': {2: 1, 3: 1}, # x21, x22
                'y': {0: 1},        # y11
                'w': {2: 1, 3: -1} # z21, z22
            },
            # M3
            {
                'x': {0: 1},        # x11
                'y': {1: 1, 3: -1}, # y12, y22
                'w': {1: 1, 3: 1}  # z12, z22
            },
            # M4
            {
                'x': {3: 1},        # x22
                'y': {2: 1, 0: -1}, # y21, y11
                'w': {0: 1, 2: 1}  # z11, z21
            },
            # M5
            {
                'x': {0: 1, 1: 1},  # x11, x12
                'y': {3: 1},        # y22
                'w': {0: -1, 1: 1} # z11, z12
            },
            # M6
            {
                'x': {2: 1, 0: -1}, # x21, x11
                'y': {0: 1, 1: 1},  # y11, y12
                'w': {3: 1}         # z22
            },
            # M7
            {
                'x': {1: 1, 3: -1}, # x12, x22
                'y': {2: 1, 3: 1},  # y21, y22
                'w': {0: 1}         # z11
            }
        ]
        
        for t in terms:
            # Construct U, V, W vectors for this term
            # U index: base_x + local_x
            # V index: base_y + local_y
            # W index: base_z + local_w
            
            # x local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            x_map = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
            y_map = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
            w_map = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
            
            for lx, cx in t['x'].items():
                rx, cx_off = x_map[lx]
                global_u = (base_x[0] + rx) * m + (base_x[1] + cx_off)
                U[global_u, col] = cx
            
            for ly, cy in t['y'].items():
                ry, cy_off = y_map[ly]
                global_v = (base_y[0] + ry) * p + (base_y[1] + cy_off)
                V[global_v, col] = cy
            
            for lw, cw in t['w'].items():
                rw, cw_off = w_map[lw]
                global_w = (base_z[0] + rw) * n + (base_z[1] + cw_off)
                W[global_w, col] = cw
            
            col += 1

    # Define the 7 block multiplications for 4x4x4
    # A blocks: A11(0,0), A12(0,2), A21(2,0), A22(2,2)
    # B blocks: B11(0,0), B12(0,2), B21(2,0), B22(2,2)
    # C blocks: C11(0,0), C12(0,2), C21(2,0), C22(2,2)
    
    # P1 = (A11+A22)(B11+B22)
    x1 = {0: 1, 1: 1, 2: 1, 3: 1} # This is wrong. x_coeffs should map local 2x2 indices to global A indices.
    # We need to pass the linear combination of A blocks to the helper.
    # The helper expects x_coeffs for the local 2x2 matrix X.
    # But X is formed by A blocks.
    # So we need to expand X in terms of A.
    
    # Let's modify the approach.
    # Instead of passing x_coeffs, we pass the definition of X as a linear combination of A blocks.
    # X = sum_{block} coeff * A_block.
    # Then for each local entry (r,c) of X, the global A indices are determined.
    
    # Let's rewrite add_strassen_2x2 to take block definitions.
    
    # Actually, it's simpler to just compute the global U, V, W vectors directly for each of the 49 terms.
    
    # Let's do it manually for the 7 P terms.
    
    # P1 = (A11+A22)(B11+B22)
    # X = A11 + A22
    # Y = B11 + B22
    # Z = C11, C22 (with coeffs from Strassen)
    
    # We will create a function that takes:
    # x_blocks: dict {block_id: coeff} where block_id is (r_off, c_off) in 2x2 grid of blocks
    # y_blocks: dict {block_id: coeff}
    # z_blocks: dict {block_id: coeff}
    # And the Strassen term index (0-6)
    
    # This is getting complicated. Let's just hardcode the 49 terms.
    # Or better, generate them programmatically.
    
    # Let's restart the generation logic.
    
    # Clear U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    # Strassen 2x2x2 terms
    # x_idx, y_idx, w_idx are local 0..3
    # w_idx corresponds to z entry.
    # The term is: (sum cx[x] * X[x]) * (sum cy[y] * Y[y]) -> sum cw[w] * Z[w]
    
    strassen_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # Block definitions for P1..P7
    # X = sum cx * A_block
    # Y = sum cy * B_block
    # Z = sum cw * C_block
    
    # A blocks: 11(0,0), 12(0,2), 21(2,0), 22(2,2)
    # B blocks: 11(0,0), 12(0,2), 21(2,0), 22(2,2)
    # C blocks: 11(0,0), 12(0,2), 21(2,0), 22(2,2)
    
    p_defs = [
        # P1 = (A11+A22)(B11+B22)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # Note: Z coeffs are 1 for C11, C22?
        # Wait, the Z coeffs in strassen_terms are for the local 2x2 Z.
        # The block Z is the result of the block multiplication.
        # So for P1, the result is added to C11 and C22.
        # So the Z block coeffs are {0: 1, 3: 1} (C11, C22).
        
        # P2 = (A21+A22)B11
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}), # Result in C21, C22
        
        # P3 = A11(B12-B22)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}), # Result in C12, C22
        
        # P4 = A22(B21-B11)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}), # Result in C11, C21
        
        # P5 = (A11+A12)B22
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}), # Result in C11, C12
        
        # P6 = (A21-A11)(B11+B12)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # Result in C22
        
        # P7 = (A12-A22)(B21+B22)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})  # Result in C11
    ]
    
    # Map block id to offset
    block_offsets = {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)}
    
    for p_idx in range(7):
        x_blocks, y_blocks, z_blocks = p_defs[p_idx]
        
        # For each of the 7 Strassen sub-terms
        for s_idx in range(7):
            x_local, y_local, w_local = strassen_terms[s_idx]
            
            # Construct U vector
            for lx, cx in x_local.items():
                # lx is local 2x2 index in X
                # X is formed by x_blocks
                # X[lx] = sum_{b} x_blocks[b] * A_block[b][lx]
                # So for each block b with coeff cb:
                for b, cb in x_blocks.items():
                    # A_block[b] is at offset block_offsets[b]
                    # lx corresponds to local (r,c) in 2x2
                    lr, lc = divmod(lx, 2)
                    br, bc = block_offsets[b]
                    gr = br + lr
                    gc = bc + lc
                    global_u = gr * m + gc
                    U[global_u, col] += cx * cb
            
            # Construct V vector
            for ly, cy in y_local.items():
                for b, cb in y_blocks.items():
                    lr, lc = divmod(ly, 2)
                    br, bc = block_offsets[b]
                    gr = br + lr
                    gc = bc + lc
                    global_v = gr * p + gc
                    V[global_v, col] += cy * cb
            
            # Construct W vector
            for lw, cw in w_local.items():
                for b, cb in z_blocks.items():
                    lr, lc = divmod(lw, 2)
                    br, bc = block_offsets[b]
                    gr = br + lr
                    gc = bc + lc
                    global_w = gr * n + gc
                    W[global_w, col] += cw * cb
            
            col += 1
    
    # Add M_4,4,1 terms (16 terms)
    for i in range(4):
        for j in range(4):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
