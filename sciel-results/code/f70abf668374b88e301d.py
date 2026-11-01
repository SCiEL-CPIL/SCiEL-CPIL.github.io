"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * (p/2) = 7 * 6 = 42
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    A_coeffs = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [0, 1, 0, -1], # M6: A01 - A11
        [-1, 0, 1, 0], # M7: A10 - A00
    ]
    
    # Strassen coefficients for B block (2x2)
    # Indices in block: 00, 01, 10, 11
    # 00 -> row 0, col 0
    # 01 -> row 0, col 1
    # 10 -> row 1, col 0
    # 11 -> row 1, col 1
    B_coeffs = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 0, 1, -1], # M3: B10 - B11
        [-1, 1, 0, 0], # M4: B01 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1], # M7: B11 + B10
    ]
    
    # Strassen coefficients for C block (2x2)
    # 00, 01, 10, 11
    C_coeffs = [
        [1, 0, 0, 1], # M1: C00 + C11
        [0, 0, 1, -1], # M2: C10 - C11  (C21 - C22 in 1-indexed)
        [0, 1, 0, 1], # M3: C01 + C11
        [1, 0, 1, 0], # M4: C00 + C10
        [-1, 1, 0, 0], # M5: -C00 + C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0], # M7: C00
    ]
    
    num_blocks = p // 2 # 6 blocks
    
    for block_idx in range(num_blocks):
        # Base column index in B and C for this block
        # B is 2 x 12. Block uses cols 2*block_idx and 2*block_idx+1
        # Row 0 of B: indices 2*block_idx, 2*block_idx+1
        # Row 1 of B: indices 12 + 2*block_idx, 12 + 2*block_idx+1
        b_base = 2 * block_idx
        # C is 12 x 2. Block uses cols 2*block_idx and 2*block_idx+1 of C?
        # C is p x n = 12 x 2.
        # W[k*n+i]. k is row of C (0..11), i is col of C (0..1).
        # Block idx corresponds to C columns? No.
        # A (2x2) * B (2x12) = C (2x12).
        # C row 0 corresponds to C[0, :] in standard notation.
        # In W vector, index is k*n + i.
        # C[0, col] -> index col.
        # C[1, col] -> index 12 + col.
        # So for block_idx, we use C columns 2*block_idx and 2*block_idx+1.
        c_base = 2 * block_idx
        
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            
            # Fill U
            for a_idx in range(4):
                if A_coeffs[term_idx][a_idx] != 0:
                    U[a_idx][col] = A_coeffs[term_idx][a_idx]
                    
            # Fill V
            # B block indices: 00, 01, 10, 11
            # 00: B[0, b_base] -> index b_base
            # 01: B[0, b_base+1] -> index b_base+1
            # 10: B[1, b_base] -> index 12 + b_base
            # 11: B[1, b_base+1] -> index 12 + b_base+1
            b_indices = [b_base, b_base+1, 12+b_base, 12+b_base+1]
            for b_idx in range(4):
                if B_coeffs[term_idx][b_idx] != 0:
                    V[b_indices[b_idx]][col] = B_coeffs[term_idx][b_idx]
                    
            # Fill W
            # C block indices: 00, 01, 10, 11
            # 00: C[0, c_base] -> index c_base
            # 01: C[0, c_base+1] -> index c_base+1
            # 10: C[1, c_base] -> index 12 + c_base
            # 11: C[1, c_base+1] -> index 12 + c_base+1
            c_indices = [c_base, c_base+1, 12+c_base, 12+c_base+1]
            for c_idx in range(4):
                if C_coeffs[term_idx][c_idx] != 0:
                    W[c_indices[c_idx]][col] = C_coeffs[term_idx][c_idx]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
