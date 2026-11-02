"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize U, V, W
    # U: (n*m, rank) -> (16, 49)
    # V: (m*p, rank) -> (16, 49)
    # W: (p*n, rank) -> (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 definitions
    # Indices for 2x2 blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # The 7 Strassen terms for 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We will use a recursive approach or explicit 49-term definition.
    # Explicit 49-term definition is safer to avoid recursion depth or overhead.
    
    # Helper to map local 2x2 block index (0..3) to global 4x4 indices
    # Block 0: rows 0-1, cols 0-1
    # Block 1: rows 0-1, cols 2-3
    # Block 2: rows 2-3, cols 0-1
    # Block 3: rows 2-3, cols 2-3
    
    def block_to_global(block_id, local_r, local_c):
        br, bc = divmod(block_id, 2)
        return br * 2 + local_r, bc * 2 + local_c

    # Define the 7 base Strassen products
    # Each product is defined by:
    # A_part: list of (block_id, coeff)
    # B_part: list of (block_id, coeff)
    # C_part: list of (block_id, coeff)
    
    # Note: The "block_id" here refers to the 2x2 blocks of the 4x4 matrix.
    # However, Strassen is recursive.
    # Level 1: 4x4 is made of four 2x2 blocks.
    # We apply Strassen to these 2x2 blocks?
    # No, Strassen multiplies 2x2 matrices.
    # To multiply 4x4 matrices, we can view them as 2x2 matrices of 2x2 blocks.
    # Let A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # Let B = [[B00, B01], [B10, B11]] where Bij are 2x2.
    # C = A * B.
    # Using Strassen on the 2x2 "super-matrices":
    # M1 = (A00 + A11) * (B00 + B11)
    # ...
    # Each M_k is a 2x2 matrix product of 2x2 blocks.
    # So we need to decompose each 2x2 block multiplication into rank 7.
    # Total rank = 7 * 7 = 49.
    
    # For each of the 7 outer Strassen terms (M1..M7), we have a product of two 2x2 matrices.
    # Let X and Y be the 2x2 matrices being multiplied.
    # We decompose X*Y into 7 rank-1 terms (P1..P7).
    # So for each outer term t (1..7) and inner term s (1..7), we have one rank-1 component.
    
    # Outer Term Definitions (for 2x2 block matrices):
    # A_comb: list of (block_idx, coeff)
    # B_comb: list of (block_idx, coeff)
    # C_comb: list of (block_idx, coeff)
    
    outer_terms = [
        # M1: (A00+A11)(B00+B11)
        {
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, -1), (1, 1), (2, 1)] # C00 += M1, C11 += -M1? 
            # Standard Strassen Output:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
        },
        # M2: (A10+A11)B00
        {
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1), (3, -1)]
        },
        # M3: A00(B01-B11)
        {
            'A': [(0, 1)],
            'B': [(1, 1), (3, -1)],
            'C': [(1, 1), (3, 1)]
        },
        # M4: A11(B10-B00)
        {
            'A': [(3, 1)],
            'B': [(2, 1), (0, -1)],
            'C': [(0, 1), (2, 1)]
        },
        # M5: (A00+A01)B11
        {
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(0, -1), (1, 1)]
        },
        # M6: (A10-A00)(B00+B01)
        {
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(3, 1)]
        },
        # M7: (A01-A11)(B10+B11)
        {
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)],
            'C': [(0, 1)]
        }
    ]
    
    # Inner Strassen Terms (for 2x2 scalar matrices):
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # P1 = (x00+x11)(y00+y11)
    # P2 = (x10+x11)y00
    # P3 = x00(y01-y11)
    # P4 = x11(y10-y00)
    # P5 = (x00+x01)y11
    # P6 = (x10-x00)(y00+y01)
    # P7 = (x01-x11)(y10+y11)
    
    # Output of inner Strassen:
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    inner_terms = [
        # P1
        {
            'X': [((0,0), 1), ((1,1), 1)],
            'Y': [((0,0), 1), ((1,1), 1)],
            'Z': [((0,0), 1), ((1,1), 1)]
        },
        # P2
        {
            'X': [((1,0), 1), ((1,1), 1)],
            'Y': [((0,0), 1)],
            'Z': [((1,0), 1), ((1,1), -1)]
        },
        # P3
        {
            'X': [((0,0), 1)],
            'Y': [((0,1), 1), ((1,1), -1)],
            'Z': [((0,1), 1), ((1,1), 1)]
        },
        # P4
        {
            'X': [((1,1), 1)],
            'Y': [((1,0), 1), ((0,0), -1)],
            'Z': [((0,0), 1), ((1,0), 1)]
        },
        # P5
        {
            'X': [((0,0), 1), ((0,1), 1)],
            'Y': [((1,1), 1)],
            'Z': [((0,0), -1), ((0,1), 1)]
        },
        # P6
        {
            'X': [((1,0), 1), ((0,0), -1)],
            'Y': [((0,0), 1), ((0,1), 1)],
            'Z': [((1,1), 1)]
        },
        # P7
        {
            'X': [((0,1), 1), ((1,1), -1)],
            'Y': [((1,0), 1), ((1,1), 1)],
            'Z': [((0,0), 1)]
        }
    ]
    
    # Construct U, V, W
    for t_idx, outer in enumerate(outer_terms):
        for s_idx, inner in enumerate(inner_terms):
            rank_idx = t_idx * 7 + s_idx
            
            # Process U (A side)
            # X is formed by linear combination of A blocks.
            # X[r,c] = sum_{(b, c_b) in outer['A']} c_b * A_block_b[r,c]
            # The inner term 'X' specifies which local (r,c) of X are used and with what coeff.
            # So for each (x_r, x_c, x_coeff) in inner['X']:
            #    For each (b, b_coeff) in outer['A']:
            #        The element A_block_b[x_r, x_c] contributes x_coeff * b_coeff to U.
            #        Global index of A_block_b[x_r, x_c] is:
            #           br, bc = divmod(b, 2)
            #           i = br*2 + x_r
            #           j = bc*2 + x_c
            #           u_idx = i*m + j
            
            U_contrib = {}
            for (x_r, x_c), x_coeff in inner['X']:
                for b, b_coeff in outer['A']:
                    br, bc = divmod(b, 2)
                    i = br * 2 + x_r
                    j = bc * 2 + x_c
                    u_idx = i * m + j
                    coeff = x_coeff * b_coeff
                    U_contrib[u_idx] = U_contrib.get(u_idx, 0) + coeff
            
            for u_idx, coeff in U_contrib.items():
                if coeff != 0:
                    U[u_idx][rank_idx] = coeff
            
            # Process V (B side)
            # Y is formed by linear combination of B blocks.
            V_contrib = {}
            for (y_r, y_c), y_coeff in inner['Y']:
                for b, b_coeff in outer['B']:
                    br, bc = divmod(b, 2)
                    i = br * 2 + y_r
                    j = bc * 2 + y_c
                    v_idx = i * p + j
                    coeff = y_coeff * b_coeff
                    V_contrib[v_idx] = V_contrib.get(v_idx, 0) + coeff
            
            for v_idx, coeff in V_contrib.items():
                if coeff != 0:
                    V[v_idx][rank_idx] = coeff
            
            # Process W (C side)
            # Z is the result of the inner product.
            # Z[r,c] contributes to C blocks.
            # C_block_b[z_r, z_c] gets contribution from Z[z_r, z_c].
            # The outer term 'C' specifies which C blocks are affected and with what coeff.
            # So for each (z_r, z_c, z_coeff) in inner['Z']:
            #    For each (b, b_coeff) in outer['C']:
            #        The element C_block_b[z_r, z_c] gets z_coeff * b_coeff.
            #        Global index of C_block_b[z_r, z_c] is:
            #           br, bc = divmod(b, 2)
            #           i = br*2 + z_r
            #           j = bc*2 + z_c
            #           w_idx = j*n + i  (Trace convention: W[k*n+i, r] where i=row, k=col)
            
            W_contrib = {}
            for (z_r, z_c), z_coeff in inner['Z']:
                for b, b_coeff in outer['C']:
                    br, bc = divmod(b, 2)
                    i = br * 2 + z_r
                    j = bc * 2 + z_c
                    w_idx = j * n + i
                    coeff = z_coeff * b_coeff
                    W_contrib[w_idx] = W_contrib.get(w_idx, 0) + coeff
            
            for w_idx, coeff in W_contrib.items():
                if coeff != 0:
                    W[w_idx][rank_idx] += coeff # Use += in case of multiple contributions? 
                    # Actually, each rank_idx is unique to this (t,s) pair.
                    # So simple assignment is fine, but += is safer if logic overlaps.
                    # Here, rank_idx is unique, so no overlap from other terms.
                    # But within this term, could multiple (b, z) pairs map to same w_idx?
                    # Yes, if different blocks b map to same global i,j? No, blocks are disjoint.
                    # So simple assignment is fine.
                    W[w_idx][rank_idx] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
