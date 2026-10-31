import time
import numpy as np

def construct():
    """
    Returns a rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = 4, 4, 4
    
    # Rank 49 is optimal for 4x4 matrix multiplication
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # We represent linear combinations of matrix elements as dictionaries.
    # Key: (row, col), Value: coefficient
    # For A (4x4), index in U is row * 4 + col
    # For B (4x4), index in V is row * 4 + col
    # For C (4x4), index in W is col * 4 + row (based on prompt W[k*n+i])
    
    def add_dict(d1, d2):
        res = dict(d1)
        for k, v in d2.items():
            res[k] = res.get(k, 0) + v
        # Remove zeros
        return {k: v for k, v in res.items() if v != 0}
        
    def sub_dict(d1, d2):
        res = dict(d1)
        for k, v in d2.items():
            res[k] = res.get(k, 0) - v
        return {k: v for k, v in res.items() if v != 0}

    # Define the 2x2 Strassen decomposition for blocks.
    # We are splitting 4x4 into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where A00 is A[0:2, 0:2], etc.
    # B = [[B00, B01], [B10, B11]]
    # C = A B
    
    # Strassen terms for the outer 2x2 block multiplication:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We need to express these M's in terms of the original 4x4 elements.
    # Each M is a 2x2 matrix product of 2x2 matrices.
    # We will apply Strassen again inside each M (Inner Strassen).
    # Inner Strassen for 2x2 matrices X, Y:
    # P1 = (X00+X11)(Y00+Y11)
    # P2 = (X10+X11)Y00
    # P3 = X00(Y01-Y11)
    # P4 = X11(Y10-Y00)
    # P5 = (X00+X01)Y11
    # P6 = (X10-X00)(Y00+Y01)
    # P7 = (X01-X11)(Y10+Y11)
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # We will iterate over the 7 outer terms and 7 inner terms.
    
    # Helper to get a 2x2 block of A as a list of 4 dicts (elements)
    # A_block[br][bc] is a 2x2 matrix of dicts? No, let's just work with the 4 elements.
    # Let's define the 4 elements of a 2x2 block as a tuple of 4 dicts: (top-left, top-right, bottom-left, bottom-right)
    
    def get_A_block(br, bc):
        # br, bc in {0, 1}
        # Elements are A[2*br + i, 2*bc + j] for i,j in 0,1
        tl = {(2*br, 2*bc): 1}
        tr = {(2*br, 2*bc + 1): 1}
        bl = {(2*br + 1, 2*bc): 1}
        br_el = {(2*br + 1, 2*bc + 1): 1}
        return (tl, tr, bl, br_el)
        
    def get_B_block(br, bc):
        tl = {(2*br, 2*bc): 1}
        tr = {(2*br, 2*bc + 1): 1}
        bl = {(2*br + 1, 2*bc): 1}
        br_el = {(2*br + 1, 2*bc + 1): 1}
        return (tl, tr, bl, br_el)
        
    # Combine two 2x2 blocks (represented as 4 dicts) with operations
    def block_add(b1, b2):
        return (add_dict(b1[0], b2[0]), add_dict(b1[1], b2[1]), add_dict(b1[2], b2[2]), add_dict(b1[3], b2[3]))
        
    def block_sub(b1, b2):
        return (sub_dict(b1[0], b2[0]), sub_dict(b1[1], b2[1]), sub_dict(b1[2], b2[2]), sub_dict(b1[3], b2[3]))
        
    # Get specific elements of a block for inner Strassen
    # X = (x00, x01, x10, x11)
    def inner_strassen_factors(X, Y):
        # X, Y are tuples of 4 dicts
        # Returns 7 pairs of (X_factor, Y_factor) where each factor is a single dict (scalar linear combo)
        # P1 = (X00+X11)(Y00+Y11)
        x1 = add_dict(X[0], X[3])
        y1 = add_dict(Y[0], Y[3])
        
        # P2 = (X10+X11)Y00
        x2 = add_dict(X[2], X[3])
        y2 = Y[0]
        
        # P3 = X00(Y01-Y11)
        x3 = X[0]
        y3 = sub_dict(Y[1], Y[3])
        
        # P4 = X11(Y10-Y00)
        x4 = X[3]
        y4 = sub_dict(Y[2], Y[0])
        
        # P5 = (X00+X01)Y11
        x5 = add_dict(X[0], X[1])
        y5 = Y[3]
        
        # P6 = (X10-X00)(Y00+Y01)
        x6 = sub_dict(X[2], X[0])
        y6 = add_dict(Y[0], Y[1])
        
        # P7 = (X01-X11)(Y10+Y11)
        x7 = sub_dict(X[1], X[3])
        y7 = add_dict(Y[2], Y[3])
        
        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7)]
        
    # Outer blocks
    A00 = get_A_block(0, 0)
    A01 = get_A_block(0, 1)
    A10 = get_A_block(1, 0)
    A11 = get_A_block(1, 1)
    
    B00 = get_B_block(0, 0)
    B01 = get_B_block(0, 1)
    B10 = get_B_block(1, 0)
    B11 = get_B_block(1, 1)
    
    # Outer Strassen terms (X, Y)
    # Each is a pair of 2x2 blocks (tuples of 4 dicts)
    outer_terms = [
        (block_add(A00, A11), block_add(B00, B11)), # M1
        (block_add(A10, A11), B00),                 # M2
        (A00, block_sub(B01, B11)),                 # M3
        (A11, block_sub(B10, B00)),                 # M4
        (block_add(A00, A01), B11),                 # M5
        (block_sub(A10, A00), block_add(B00, B01)), # M6
        (block_sub(A01, A11), block_add(B10, B11))  # M7
    ]
    
    # Contribution of each outer term M_t to the C blocks (C00, C01, C10, C11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # We represent C blocks as 2x2 matrices of "target" positions in the final 4x4 C.
    # C00 covers C[0:2, 0:2], C01 covers C[0:2, 2:4], etc.
    # For a given outer term t, and an inner result position (li, lj) in the 2x2 block,
    # the global position in C is (br*2+li, bc*2+lj) where (br, bc) is the block position.
    
    # inner_targets[k] gives the coefficients for the 4 positions (00, 01, 10, 11) of the local 2x2 result Z
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    # So for each inner product P_k, we add its contribution to the appropriate Z positions.
    
    # Let's pre-compute the contribution of each inner P_k to the 4 local positions
    # local_pos 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    inner_contribs = [
        [1, 0, 0, 1],  # P1 -> Z00, Z11
        [0, 0, 1, -1], # P2 -> Z10, Z11
        [0, 1, 0, 1],  # P3 -> Z01, Z11
        [1, 0, 1, 0],  # P4 -> Z00, Z10
        [-1, 1, 0, 0], # P5 -> Z00, Z01
        [0, 0, 0, 1],  # P6 -> Z11
        [1, 0, 0, 0]   # P7 -> Z00
    ]
    
    # Outer term contributions to C blocks
    # C00: M1, M4, -M5, M7
    # C01: M3, M5
    # C10: M2, M4
    # C11: M1, -M2, M3, M6
    # We need to map (outer_term_idx, inner_pos) to (global_C_row, global_C_col) and coefficient.
    # Let's just iterate and fill W.
    
    col = 0
    for t in range(7):
        if time.time() > deadline:
             # Return whatever we have, though this shouldn't happen
             break
             
        X_block, Y_block = outer_terms[t]
        inner_factors = inner_strassen_factors(X_block, Y_block)
        
        for k in range(7):
            x_dict, y_dict = inner_factors[k]
            
            # Fill U from x_dict
            for (r, c), coeff in x_dict.items():
                U[r * 4 + c, col] += coeff
                
            # Fill V from y_dict
            for (r, c), coeff in y_dict.items():
                V[r * 4 + c, col] += coeff
                
            # Fill W
            # We need to determine which C elements are affected.
            # The inner product P_k contributes to local positions Z_li_lj with coefficient inner_contribs[k][li*2+lj]
            # The outer term M_t contributes to global C blocks (br, bc) with coefficient outer_contrib[t][br*2+bc]
            
            # Let's define outer_contribs explicitly
            # t=0 (M1): C00(+1), C11(+1)
            # t=1 (M2): C10(+1), C11(-1)
            # t=2 (M3): C01(+1), C11(+1)
            # t=3 (M4): C00(+1), C10(+1)
            # t=4 (M5): C00(-1), C01(+1)
            # t=5 (M6): C11(+1)
            # t=6 (M7): C00(+1)
            
            # We'll use a helper to get the global C indices for a given outer term and local pos
            # This is slightly complex, so let's just hardcode the mapping or compute it.
            
            # Local pos 0 (0,0) of M_t block:
            # If M_t is in block (br, bc), local (0,0) is global (br*2, bc*2)
            # The coefficient is outer_coeff * inner_coeff
            
            # Let's define the block positions and coefficients for each outer term
            # M1 (t=0): Blocks (0,0) coeff 1, (1,1) coeff 1
            # M2 (t=1): Blocks (1,0) coeff 1, (1,1) coeff -1
            # M3 (t=2): Blocks (0,1) coeff 1, (1,1) coeff 1
            # M4 (t=3): Blocks (0,0) coeff 1, (1,0) coeff 1
            # M5 (t=4): Blocks (0,0) coeff -1, (0,1) coeff 1
            # M6 (t=5): Blocks (1,1) coeff 1
            # M7 (t=6): Blocks (0,0) coeff 1
            
            outer_block_map = [
                [(0,0,1), (1,1,1)],
                [(1,0,1), (1,1,-1)],
                [(0,1,1), (1,1,1)],
                [(0,0,1), (1,0,1)],
                [(0,0,-1), (0,1,1)],
                [(1,1,1)],
                [(0,0,1)]
            ]
            
            for (br, bc, coeff_o) in outer_block_map[t]:
                # For each local position affected by inner product k
                for li in range(2):
                    for lj in range(2):
                        idx_local = li * 2 + lj
                        coeff_i = inner_contribs[k][idx_local]
                        if coeff_i != 0:
                            global_i = br * 2 + li
                            global_k = bc * 2 + lj
                            w_idx = global_k * 4 + global_i
                            W[w_idx, col] += coeff_o * coeff_i
                            
            col += 1
            
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
