"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x2 * 2x8
    # We split the 2x8 matrix into two 2x4 blocks.
    # Multiplying 2x2 by 2x4 using Strassen requires 7 scalar multiplications per 2x2 output block.
    # Since we have two 2x4 blocks, we can treat the multiplication as two separate 2x2 * 2x2 multiplications
    # if we split the 8 columns into four 2-column groups? No, Strassen works on 2x2 blocks.
    # A 2x8 matrix can be viewed as 4 blocks of 2x2.
    # So we have 4 independent 2x2 * 2x2 multiplications.
    # Each 2x2 * 2x2 multiplication can be done with rank 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=8
    # U shape: (n*m, rank) = (4, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (16, 28)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 * 2x2:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # We have 4 such blocks.
    # Block 0: A (2x2) * B0 (2x2) -> C0 (2x2)
    # Block 1: A (2x2) * B1 (2x2) -> C1 (2x2)
    # Block 2: A (2x2) * B2 (2x2) -> C2 (2x2)
    # Block 3: A (2x2) * B3 (2x2) -> C3 (2x2)
    
    # Indices:
    # A is 2x2, indices (i,j) with i in {0,1}, j in {0,1}
    # U index: i*m + j = i*2 + j
    # B is 2x8, indices (j,k) with j in {0,1}, k in {0..7}
    # V index: j*p + k = j*8 + k
    # C is 2x8, indices (i,k) with i in {0,1}, k in {0..7}
    # W index: k*n + i = k*2 + i (following trace convention W[k*n+i, r])
    
    # Strassen coefficients for a single 2x2*2x2 block:
    # For block b (0..3), the B block columns are 4*b to 4*b+3.
    # e = B[0, 4b], f = B[0, 4b+1], g = B[1, 4b], h = B[1, 4b+1]
    # Wait, B is 2x2, so:
    # B = [[e, f], [g, h]]
    # e is at (0, 4b), f at (0, 4b+1)
    # g at (1, 4b), h at (1, 4b+1)
    
    # Let's define the Strassen terms for one block.
    # Term 1: (a+d)(e+h)
    # U contributions: a (U[0,0]=1), d (U[1,1]=1)
    # V contributions: e (V[0, 4b]=1), h (V[1, 4b+1]=1)
    # W contributions:
    # C11 += M1, C12 -= M1, C21 -= M1, C22 += M1
    # C11 is at (0, 4b), C12 at (0, 4b+1), C21 at (1, 4b), C22 at (1, 4b+1)
    # W index for C11: (4b)*2 + 0 = 8b
    # W index for C12: (4b+1)*2 + 0 = 8b+2
    # W index for C21: (4b)*2 + 1 = 8b+1
    # W index for C22: (4b+1)*2 + 1 = 8b+3
    
    strassen_terms = [
        # (U_indices, V_indices, W_indices, signs)
        # Term 1: (a+d)(e+h)
        # U: 0 (a), 2 (d)  [i=0,j=0 -> 0; i=1,j=1 -> 3? No, i*m+j. i=1,j=1 -> 1*2+1=3]
        # Let's re-check U indices.
        # U[i*m+j]. m=2.
        # i=0, j=0 -> 0 (a)
        # i=0, j=1 -> 1 (b)
        # i=1, j=0 -> 2 (c)
        # i=1, j=1 -> 3 (d)
        
        # Term 1: (a+d)(e+h)
        # U: [0, 3]
        # V: [0*8+4b, 1*8+4b+1] -> [4b, 8+4b+1]
        # W: [8b, 8b+2, 8b+1, 8b+3] with signs [1, -1, -1, 1]
        (
            [0, 3],
            [4, 9],  # relative to block offset
            [0, 2, 1, 3],  # relative to block offset * 2
            [1, -1, -1, 1]
        ),
        # Term 2: (c+d)(e)
        # U: [2, 3]
        # V: [4b]
        # W: [8b+1, 8b+3] with signs [1, 1]  (C21 += M2, C22 -= M2)
        (
            [2, 3],
            [4],
            [1, 3],
            [1, -1]
        ),
        # Term 3: (a)(f-h)
        # U: [0]
        # V: [4b+1, 8+4b+1] with signs [1, -1]
        # W: [8b+2, 8b+3] with signs [1, 1]  (C12 += M3, C22 += M3)
        (
            [0],
            [5, 9],
            [2, 3],
            [1, 1]
        ),
        # Term 4: (d)(g-e)
        # U: [3]
        # V: [8+4b, 4b] with signs [1, -1]
        # W: [8b, 8b+1] with signs [-1, 1]  (C11 -= M4, C21 += M4)
        (
            [3],
            [8, 4],
            [0, 1],
            [-1, 1]
        ),
        # Term 5: (a+b)(h)
        # U: [0, 1]
        # V: [8+4b+1]
        # W: [8b, 8b+2] with signs [-1, 1]  (C11 -= M5, C12 += M5)
        (
            [0, 1],
            [9],
            [0, 2],
            [-1, 1]
        ),
        # Term 6: (c-a)(e+g)
        # U: [2, 0] with signs [1, -1]
        # V: [4b, 8+4b]
        # W: [8b+1] with signs [1]  (C21 += M6)
        (
            [2, 0],
            [4, 8],
            [1],
            [1]
        ),
        # Term 7: (b-d)(f+h)
        # U: [1, 3] with signs [1, -1]
        # V: [4b+1, 8+4b+1]
        # W: [8b, 8b+3] with signs [-1, 1]  (C11 -= M7, C22 += M7)
        (
            [1, 3],
            [5, 9],
            [0, 3],
            [-1, 1]
        ),
    ]
    
    # Note: The V indices above are relative to the block's start in the 2x4 block.
    # But V is 2x8, so for block b, the columns are 4b to 4b+3.
    # V index = j*8 + k.
    # For e (j=0, k=4b): 0*8 + 4b = 4b
    # For f (j=0, k=4b+1): 0*8 + 4b+1 = 4b+1
    # For g (j=1, k=4b): 1*8 + 4b = 8+4b
    # For h (j=1, k=4b+1): 1*8 + 4b+1 = 8+4b+1
    
    # My relative indices in the list above assumed the block starts at column 0.
    # So for block b, I need to add 4*b to the column part of the V index.
    # V index = j*8 + (k_rel + 4*b)
    
    # Let's re-construct the V indices for each term and block.
    
    for b in range(4):
        base_col = 4 * b
        for t in range(7):
            col_offset = 7 * b + t
            u_indices, v_rel_indices, w_rel_indices, signs = strassen_terms[t]
            
            # V indices
            # v_rel_indices are like [4, 9] which means j=0,k=4 and j=1,k=1?
            # No, my previous derivation was messy. Let's be precise.
            
            # e: j=0, k=0 (rel) -> abs k=base_col. V_idx = 0*8 + base_col
            # f: j=0, k=1 (rel) -> abs k=base_col+1. V_idx = 0*8 + base_col+1
            # g: j=1, k=0 (rel) -> abs k=base_col. V_idx = 1*8 + base_col
            # h: j=1, k=1 (rel) -> abs k=base_col+1. V_idx = 1*8 + base_col+1
            
            # Let's map the v_rel_indices from my previous list to actual V indices.
            # Term 1: (a+d)(e+h). V: e, h.
            # e -> 0*8 + base_col
            # h -> 1*8 + base_col + 1
            # Term 2: (c+d)(e). V: e.
            # e -> 0*8 + base_col
            # Term 3: (a)(f-h). V: f, h.
            # f -> 0*8 + base_col + 1
            # h -> 1*8 + base_col + 1
            # Term 4: (d)(g-e). V: g, e.
            # g -> 1*8 + base_col
            # e -> 0*8 + base_col
            # Term 5: (a+b)(h). V: h.
            # h -> 1*8 + base_col + 1
            # Term 6: (c-a)(e+g). V: e, g.
            # e -> 0*8 + base_col
            # g -> 1*8 + base_col
            # Term 7: (b-d)(f+h). V: f, h.
            # f -> 0*8 + base_col + 1
            # h -> 1*8 + base_col + 1
            
            # W indices
            # C11: i=0, k=base_col. W_idx = base_col*2 + 0
            # C12: i=0, k=base_col+1. W_idx = (base_col+1)*2 + 0
            # C21: i=1, k=base_col. W_idx = base_col*2 + 1
            # C22: i=1, k=base_col+1. W_idx = (base_col+1)*2 + 1
            
            # Let's hardcode the V and W indices for each term to be safe.
            
            v_indices = [
                [base_col, 8 + base_col + 1],      # T1: e, h
                [base_col],                         # T2: e
                [base_col + 1, 8 + base_col + 1],   # T3: f, h
                [8 + base_col, base_col],           # T4: g, e
                [8 + base_col + 1],                 # T5: h
                [base_col, 8 + base_col],           # T6: e, g
                [base_col + 1, 8 + base_col + 1],   # T7: f, h
            ][t]
            
            w_indices = [
                [2*base_col, 2*(base_col+1), 2*base_col+1, 2*(base_col+1)+1], # T1: C11, C12, C21, C22
                [2*base_col+1, 2*(base_col+1)+1],                              # T2: C21, C22
                [2*(base_col+1), 2*(base_col+1)+1],                            # T3: C12, C22
                [2*base_col, 2*base_col+1],                                    # T4: C11, C21
                [2*base_col, 2*(base_col+1)],                                  # T5: C11, C12
                [2*base_col+1],                                                # T6: C21
                [2*base_col, 2*(base_col+1)+1],                                # T7: C11, C22
            ][t]
            
            # Signs for W
            w_signs = [
                [1, -1, -1, 1],  # T1
                [1, -1],         # T2
                [1, 1],          # T3
                [-1, 1],         # T4
                [-1, 1],         # T5
                [1],             # T6
                [-1, 1],         # T7
            ][t]
            
            # Signs for U (for terms with 2 U indices)
            u_signs = [
                [1, 1],   # T1: a+d
                [1, 1],   # T2: c+d
                [1],      # T3: a
                [1],      # T4: d
                [1, 1],   # T5: a+b
                [1, -1],  # T6: c-a
                [1, -1],  # T7: b-d
            ][t]
            
            # Signs for V (for terms with 2 V indices)
            v_signs = [
                [1, 1],   # T1: e+h
                [1],      # T2: e
                [1, -1],  # T3: f-h
                [1, -1],  # T4: g-e
                [1],      # T5: h
                [1, 1],   # T6: e+g
                [1, 1],   # T7: f+h
            ][t]
            
            # Fill U
            for u_idx, u_sign in zip(u_indices, u_signs):
                U[u_idx][col_offset] = u_sign
                
            # Fill V
            for v_idx, v_sign in zip(v_indices, v_signs):
                V[v_idx][col_offset] = v_sign
                
            # Fill W
            for w_idx, w_sign in zip(w_indices, w_signs):
                W[w_idx][col_offset] = w_sign
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
