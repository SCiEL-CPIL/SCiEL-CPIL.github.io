"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # B is split into B_L (4x4) and B_R (4x5)
    # B_L uses Strassen 4x4x4 (rank 49)
    # B_R uses trivial 4x4x5 (rank 80)
    # Total rank 129
    rank = 129
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: Trivial decomposition for B_R (cols 4..8) ---
    col = 0
    # B_R corresponds to global k in 4..8
    # A is 4x4, B_R is 4x5
    # i in 0..3, j in 0..3, k_local in 0..4
    for i in range(n):
        for j in range(m):
            for k_local in range(5):
                k_global = k_local + 4
                U[i * m + j][col] = 1
                V[j * p + k_global][col] = 1
                W[k_global * n + i][col] = 1
                col += 1
                
    # --- Part 2: Strassen decomposition for B_L (cols 0..3) ---
    # We need to define the 7 Strassen blocks for 4x4x4
    # Each block is a 2x2x2 multiplication
    
    # Local Strassen 2x2x2 vectors (u, v, w)
    # Indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    strassen_2x2 = [
        # u, v, w
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1: (x00+x11)(y00+y11) -> z00+1, z11+1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2: (x10+x11)y00 -> z10+1, z11-1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3: x00(y01-y11) -> z01+1, z11+1
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4: x11(y10-y00) -> z00+1, z10+1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5: (x00+x01)y11 -> z00-1, z01+1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6: (x10-x00)(y00+y01) -> z11+1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7: (x01-x11)(y10+y11) -> z00+1
    ]
    
    # Define the 7 blocks of the 4x4x4 Strassen algorithm
    # Each block specifies:
    # 1. How to form X (2x2) from A blocks
    # 2. How to form Y (2x2) from B_L blocks
    # 3. How to add Z (2x2) to C_L blocks
    
    # A blocks:
    # A11: (0,0)-(1,1)
    # A12: (0,0)-(1,1) shifted cols by 2
    # A21: (0,0)-(1,1) shifted rows by 2
    # A22: (0,0)-(1,1) shifted rows and cols by 2
    
    # We represent the linear combination of A blocks to form X
    # X is 2x2. X[0,0] is sum of alpha * A_block.
    # We need to map local X indices to global A indices.
    
    # Let's define the blocks explicitly as lists of (block_id, coeff)
    # Block IDs: 0=A11, 1=A12, 2=A21, 3=A22
    # Local X index 0 -> X[0,0], 1 -> X[0,1], 2 -> X[1,0], 3 -> X[1,1]
    
    # Block 1: X = A11 + A22
    # X[0,0] = A11[0,0] + A22[0,0]
    # X[0,1] = A11[0,1] + A22[0,1]
    # X[1,0] = A11[1,0] + A22[1,0]
    # X[1,1] = A11[1,1] + A22[1,1]
    
    # This structure is identical for all blocks, just different coeffs.
    # We can define a function to get the global index for a local index in a block.
    
    def get_global_a_index(block_id, local_r, local_c):
        # block_id: 0=A11, 1=A12, 2=A21, 3=A22
        # local_r, local_c in 0,1
        if block_id == 0: # A11
            return (local_r) * m + (local_c)
        elif block_id == 1: # A12
            return (local_r) * m + (local_c + 2)
        elif block_id == 2: # A21
            return (local_r + 2) * m + (local_c)
        elif block_id == 3: # A22
            return (local_r + 2) * m + (local_c + 2)

    def get_global_b_index(block_id, local_r, local_c):
        # B_L is 4x4, cols 0..3
        # block_id: 0=B11, 1=B12, 2=B21, 3=B22
        # local_r, local_c in 0,1
        # Global B is 4x9. Rows 0..3, Cols 0..8.
        # B_L cols are 0..3.
        if block_id == 0: # B11
            return (local_r) * p + (local_c)
        elif block_id == 1: # B12
            return (local_r) * p + (local_c + 2)
        elif block_id == 2: # B21
            return (local_r + 2) * p + (local_c)
        elif block_id == 3: # B22
            return (local_r + 2) * p + (local_c + 2)

    def get_global_c_index(block_id, local_r, local_c):
        # C_L is 4x4, cols 0..3
        # block_id: 0=C11, 1=C12, 2=C21, 3=C22
        # local_r, local_c in 0,1
        # Global C is 4x9. Rows 0..3, Cols 0..8.
        # W index is k * n + i. k is col, i is row.
        # So we need global k and i.
        if block_id == 0: # C11
            i = local_r
            k = local_c
        elif block_id == 1: # C12
            i = local_r
            k = local_c + 2
        elif block_id == 2: # C21
            i = local_r + 2
            k = local_c
        elif block_id == 3: # C22
            i = local_r + 2
            k = local_c + 2
        return k * n + i

    # Define the 7 Strassen blocks for 4x4
    # Each block is a tuple of (X_blocks, Y_blocks, Z_blocks, Z_coeffs)
    # X_blocks: dict mapping local_X_idx -> list of (A_block_id, coeff)
    # This is getting complex to define generally.
    # Let's hardcode the 7 blocks.
    
    # Block 1: M1 = (A11+A22)(B11+B22) -> C11 + M1, C22 + M1
    # X = A11 + A22. Y = B11 + B22.
    # Z contributes to C11 and C22.
    
    # We will iterate r in 1..7, s in 1..7.
    # For each, we determine the coefficients for global U, V, W.
    
    blocks = [
        # (X_def, Y_def, Z_def)
        # X_def: list of 4 tuples (local_r, local_c, A_block_id, coeff)
        # Actually, it's easier to say X is formed by summing A_blocks.
        # X[0,0] = sum(coeff * A_block[0,0])
        # Let's define X as a 2x2 matrix of (block_id, coeff)
        # X = [[(0,1), (0,1)], [(0,1), (0,1)]] for A11+A22? No, A22 is block 3.
        # X = [[(0,1), (3,1)], [(0,1), (3,1)]]
        
        # Block 1: X = A11 + A22
        {
            'X': [[(0, 1), (3, 1)],
                  [(0, 1), (3, 1)]],
            'Y': [[(0, 1), (3, 1)],
                  [(0, 1), (3, 1)]],
            'Z': [(0, 1), (3, 1)] # C11 + 1*M, C22 + 1*M
        },
        # Block 2: X = A21 + A22, Y = B11
        {
            'X': [[(2, 1), (3, 1)],
                  [(2, 1), (3, 1)]],
            'Y': [[(0, 1), (0, 0)],
                  [(0, 1), (0, 0)]],
            'Z': [(2, 1), (3, -1)] # C21 + 1*M, C22 - 1*M
        },
        # Block 3: X = A11, Y = B12 - B22
        {
            'X': [[(0, 1), (0, 0)],
                  [(0, 1), (0, 0)]],
            'Y': [[(1, 1), (3, -1)],
                  [(1, 1), (3, -1)]],
            'Z': [(1, 1), (3, 1)] # C12 + 1*M, C22 + 1*M
        },
        # Block 4: X = A22, Y = B21 - B11
        {
            'X': [[(3, 1), (3, 0)],
                  [(3, 1), (3, 0)]],
            'Y': [[(2, 1), (0, -1)],
                  [(2, 1), (0, -1)]],
            'Z': [(0, 1), (2, 1)] # C11 + 1*M, C21 + 1*M
        },
        # Block 5: X = A11 + A12, Y = B22
        {
            'X': [[(0, 1), (1, 1)],
                  [(0, 1), (1, 1)]],
            'Y': [[(3, 1), (3, 0)],
                  [(3, 1), (3, 0)]],
            'Z': [(0, -1), (1, 1)] # C11 - 1*M, C12 + 1*M
        },
        # Block 6: X = A21 - A11, Y = B11 + B12
        {
            'X': [[(2, 1), (0, -1)],
                  [(2, 1), (0, -1)]],
            'Y': [[(0, 1), (1, 1)],
                  [(0, 1), (1, 1)]],
            'Z': [(3, 1)] # C22 + 1*M
        },
        # Block 7: X = A12 - A22, Y = B21 + B22
        {
            'X': [[(1, 1), (3, -1)],
                  [(1, 1), (3, -1)]],
            'Y': [[(2, 1), (3, 1)],
                  [(2, 1), (3, 1)]],
            'Z': [(0, 1)] # C11 + 1*M
        }
    ]
    
    for r in range(7):
        block = blocks[r]
        X_def = block['X']
        Y_def = block['Y']
        Z_def = block['Z']
        
        for s in range(7):
            u_s, v_s, w_s = strassen_2x2[s]
            
            # u_s is [x00, x01, x10, x11]
            # v_s is [y00, y01, y10, y11]
            # w_s is [z00, z01, z10, z11]
            
            # Accumulate contributions to global U, V, W
            # U index: i*m + j
            # V index: j*p + k
            # W index: k*n + i
            
            # Process U (from X)
            for local_idx in range(4):
                coeff_x = u_s[local_idx]
                if coeff_x == 0: continue
                lr = local_idx // 2
                lc = local_idx % 2
                # X_def[lr][lc] is a list of (block_id, coeff)
                for block_id, coeff_b in X_def[lr][lc]:
                    global_idx = get_global_a_index(block_id, lr, lc)
                    U[global_idx][col] += coeff_x * coeff_b
                    
            # Process V (from Y)
            for local_idx in range(4):
                coeff_y = v_s[local_idx]
                if coeff_y == 0: continue
                lr = local_idx // 2
                lc = local_idx % 2
                for block_id, coeff_b in Y_def[lr][lc]:
                    global_idx = get_global_b_index(block_id, lr, lc)
                    V[global_idx][col] += coeff_y * coeff_b
                    
            # Process W (from Z)
            for local_idx in range(4):
                coeff_z = w_s[local_idx]
                if coeff_z == 0: continue
                lr = local_idx // 2
                lc = local_idx % 2
                # Z_def is a list of (block_id, coeff)
                # The local Z index (lr, lc) maps to the block's local position.
                # We need to find which global block this local position belongs to?
                # No, Z_def specifies which global C blocks receive contributions.
                # The local Z matrix is the result of the 2x2 multiplication.
                # The entries of Z are added to the specified C blocks.
                # Wait, Z_def is a list of (C_block_id, coeff).
                # This implies that the ENTIRE Z matrix (all 4 entries) is added to these blocks?
                # No, that's not right.
                # In Strassen, M1 is a 2x2 matrix.
                # C11 = M1 + ...
                # This means C11[0,0] += M1[0,0], C11[0,1] += M1[0,1], etc.
                # So for each local Z entry (lr, lc), we add it to the corresponding position in the global C blocks listed in Z_def.
                
                for block_id, coeff_c in Z_def:
                    global_idx = get_global_c_index(block_id, lr, lc)
                    W[global_idx][col] += coeff_z * coeff_c
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
