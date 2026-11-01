"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 Strassen-based decomposition for 2x2 * 2x12 matrix multiplication.
    Partition the 12 columns into 6 pairs, apply 7-product Strassen to each pair.
    """
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (4, 42)
    # V: (m*p, rank) = (24, 42)
    # W: (p*n, rank) = (24, 42)
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 formulas for M1..M7
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+c)h
    # M6 = (c-a)(e+g)
    # M7 = (d-c)(f+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block b (0..5), columns are 2b and 2b+1
    # B block: [[B0, B1], [B2, B3]] where B0=B[0,2b], B1=B[0,2b+1], B2=B[1,2b], B3=B[1,2b+1]
    # In flattened V: row 0 is row 0 of B (cols 0..11), row 1 is row 1 of B (cols 0..11)
    # V[0, 2b] = B[0,2b] = e
    # V[0, 2b+1] = B[0,2b+1] = f
    # V[1, 2b] = B[1,2b] = g
    # V[1, 2b+1] = B[1,2b+1] = h
    
    # U rows: 0=a=A[0,0], 1=b=A[0,1], 2=c=A[1,0], 3=d=A[1,1]
    # W rows: k*n+i where k is column index (0..11), i is row of C (0,1)
    # W[k, i] contributes to C[i, k]
    # C[0, 2b] gets contributions from M1, M4, M5, M7
    # C[0, 2b+1] gets contributions from M3, M5
    # C[1, 2b] gets contributions from M2, M4
    # C[1, 2b+1] gets contributions from M1, M2, M3, M6
    
    # Define the 7 Strassen terms for a single 2x2 * 2x2 block
    # Each term: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict mapping row index (0,1,2,3) to coefficient
    # V_coeffs: dict mapping (row, col_in_block) to coefficient, col_in_block 0 or 1
    # W_coeffs: dict mapping (row_of_C, col_in_block) to coefficient
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        # M2 = (c+d)e
        ({2: 1, 3: 1}, {(0, 0): 1}, {(1, 0): 1}),
        # M3 = a(f-h)
        ({0: 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1}),
        # M4 = d(g-e)
        ({3: 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        # M5 = (a+c)h
        ({0: 1, 2: 1}, {(1, 1): 1}, {(0, 1): -1, (1, 1): 1}),
        # M6 = (c-a)(e+g)
        ({2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}, {(1, 1): 1}),
        # M7 = (d-c)(f+h)
        ({3: 1, 2: -1}, {(0, 1): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    
    col = 0
    for b in range(6):
        for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
            # Fill U
            for row, coeff in u_coeffs.items():
                U[row][col] = coeff
            # Fill V
            for (v_row, v_col_in_block), coeff in v_coeffs.items():
                actual_col = 2 * b + v_col_in_block
                V[v_row * p + actual_col][col] = coeff
            # Fill W
            for (w_row_of_C, w_col_in_block), coeff in w_coeffs.items():
                actual_col = 2 * b + w_col_in_block
                w_index = actual_col * n + w_row_of_C
                W[w_index][col] = coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
