"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 matrix multiplication applied to 7 blocks of 2x2.
    # Total rank: 7 * 7 = 49.
    n, m, p = DIMENSIONS
    # p=14, so 7 blocks of 2 columns.
    num_blocks = p // 2
    
    # Strassen coefficients for A
    # A = [[a, b], [c, d]]
    # M1 = (a+d)(e+f)
    # M2 = (c+d)g
    # M3 = a(e-f)
    # M4 = d(g-h)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Indices for U (A part): 2x2 = 4 entries.
    # We map a->0, b->1, c->2, d->3.
    
    # Indices for V (B part): 2x2 per block = 4 entries per block.
    # Total 7 * 4 = 28 entries.
    # e->0, f->1, g->2, h->3 for block k.
    
    # Indices for W (C part): 2x2 per block = 4 entries per block.
    # Total 7 * 4 = 28 entries.
    # C11->0, C12->1, C21->2, C22->3 for block k.
    
    rank = 7 * num_blocks
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definition:
    # (coeff_A, coeff_B, coeff_C)
    # A indices: a=0, b=1, c=2, d=3
    # B indices (local): e=0, f=1, g=2, h=3
    # C indices (local): C11=0, C12=1, C21=2, C22=3
    
    # M1: (a+d)(e+f) -> C11 += M1, C22 += M1
    # M2: (c+d)g     -> C21 += M2, C22 += M2
    # M3: a(e-f)     -> C11 += M3, C12 += M3
    # M4: d(g-h)     -> C21 += M4, C22 += M4
    # M5: (a+b)h     -> C12 += M5, C22 += M5
    # M6: (c-a)(e+f) -> C21 += M6, C11 += -M6
    # M7: (b-d)(g+h) -> C12 += M7, C21 += -M7
    
    terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict {idx: val}
        # B_coeffs: dict {idx: val}
        # C_coeffs: dict {idx: val}
        
        # M1
        ({0: 1, 3: 1}, {0: 1, 1: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {2: 1}, {2: 1, 3: 1}),
        # M3
        ({0: 1}, {0: 1, 1: -1}, {0: 1, 1: 1}),
        # M4
        ({3: 1}, {2: 1, 3: -1}, {2: 1, 3: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 3: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {2: 1, 0: -1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {1: 1, 2: -1}),
    ]
    
    col = 0
    for k in range(num_blocks):
        # B global offset
        b_off = k * 2
        # C global offset
        c_off = k * 2
        
        for A_coeffs, B_coeffs, C_coeffs in terms:
            # Fill U
            for a_idx, val in A_coeffs.items():
                U[a_idx][col] = val
            
            # Fill V
            for b_idx, val in B_coeffs.items():
                # B row 0: e, f. B row 1: g, h.
                # b_idx 0,1 are row 0. 2,3 are row 1.
                if b_idx < 2:
                    row = 0
                    col_in_row = b_idx
                else:
                    row = 1
                    col_in_row = b_idx - 2
                
                # B is m x p = 2 x 14.
                # Index in V: row * p + col
                # col = b_off + col_in_row
                V[row * p + (b_off + col_in_row)][col] = val
            
            # Fill W
            for c_idx, val in C_coeffs.items():
                # C is p x n = 14 x 2.
                # W index: c_global * n + i
                # c_idx 0,1 are row 0 of C block. 2,3 are row 1 of C block.
                if c_idx < 2:
                    row = 0
                    col_in_row = c_idx
                else:
                    row = 1
                    col_in_row = c_idx - 2
                
                # C global col index: c_off + col_in_row
                # W index: (c_off + col_in_row) * n + row
                W[(c_off + col_in_row) * n + row][col] = val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
