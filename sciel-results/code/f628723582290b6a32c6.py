"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # Rows of U/V/W correspond to indices 00, 01, 10, 11 of the 2x2 matrix
    # Cols correspond to the 7 Strassen products
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 0, 0, 1],   # A01
        [0, 1, 0, 0, 1, 0, -1],  # A10
        [1, 1, 0, 1, 0, 1, -1]   # A11
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    # Total rank: 49 (for 4x4x4) + 28 (for 4x4x2) = 77
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(col_offset, a_indices, b_indices, c_indices):
        """
        Add a 2x2x2 Strassen decomposition to the global matrices.
        a_indices: list of 4 global row indices for U
        b_indices: list of 4 global row indices for V
        c_indices: list of 4 global row indices for W
        """
        for local_i in range(4):
            for r in range(7):
                val_u = U_strassen[local_i][r]
                val_v = V_strassen[local_i][r]
                val_w = W_strassen[local_i][r]
                
                if val_u != 0:
                    U[a_indices[local_i]][col_offset + r] += val_u
                if val_v != 0:
                    V[b_indices[local_i]][col_offset + r] += val_v
                if val_w != 0:
                    W[c_indices[local_i]][col_offset + r] += val_w

    # --- Part 1: 4x4x4 (Rank 49) ---
    # A is 4x4, B is 4x4 (cols 0-3 of B), C is 4x4 (cols 0-3 of C)
    
    # A block indices (i*m+j)
    # A00: i in 0,1; j in 0,1 -> 0,1,4,5
    # A01: i in 0,1; j in 2,3 -> 2,3,6,7
    # A10: i in 2,3; j in 0,1 -> 8,9,12,13
    # A11: i in 2,3; j in 2,3 -> 10,11,14,15
    
    a00 = [0, 2, 4, 6]
    a01 = [1, 3, 5, 7]
    a10 = [8, 10, 12, 14]
    a11 = [9, 11, 13, 15]
    
    # B block indices (j*p+k)
    # B00: j in 0,1; k in 0,1 -> 0,1,6,7
    # B01: j in 0,1; k in 2,3 -> 2,3,8,9
    # B10: j in 2,3; k in 0,1 -> 12,13,18,19
    # B11: j in 2,3; k in 2,3 -> 14,15,20,21
    
    b00 = [0, 2, 6, 8]
    b01 = [1, 3, 7, 9]
    b10 = [12, 14, 18, 20]
    b11 = [13, 15, 19, 21]
    
    # C block indices (k*n+i)
    # C00: k in 0,1; i in 0,1 -> 0,1,4,5
    # C01: k in 0,1; i in 2,3 -> 2,3,6,7
    # C10: k in 2,3; i in 0,1 -> 8,9,12,13
    # C11: k in 2,3; i in 2,3 -> 10,11,14,15
    
    c00 = [0, 2, 4, 6]
    c01 = [1, 3, 5, 7]
    c10 = [8, 10, 12, 14]
    c11 = [9, 11, 13, 15]
    
    col = 0
    # M1 = (A00+A11)(B00+B11)
    add_strassen(col, a00, b00, c00); col += 7
    add_strassen(col, a11, b11, c00); col += 7
    # Note: The above adds two separate Strassen decompositions.
    # But Strassen M1 is a single product of sums.
    # My add_strassen adds a full 2x2x2 multiplication.
    # So for M1, I need to compute (A00+A11) * (B00+B11).
    # This is NOT simply adding two Strassen decompositions.
    # I need to form the matrices A_sum = A00+A11 and B_sum = B00+B11.
    # But I cannot easily form them in the U/V/W matrices without extra rank.
    # 
    # Wait, the Strassen algorithm for 4x4 uses 7 products of 2x2 matrices.
    # Each product is computed using the 2x2 Strassen algorithm (rank 7).
    # So yes, for each of the 7 "big" products, we invoke the 2x2 Strassen logic.
    # But the inputs to these 2x2 Strassen calls are linear combinations of the 2x2 blocks.
    # 
    # Example: M1 = (A00+A11)(B00+B11).
    # The U matrix for this product must contain the coefficients for A00 and A11.
    # Specifically, the "A" part of the 2x2 Strassen for M1 sees the matrix A_sum.
    # A_sum = A00 + A11.
    # So the U rows for A_sum are the sum of the U rows for A00 and A11?
    # No, the U matrix maps the global A indices to the local 2x2 Strassen inputs.
    # 
    # Let's refine add_strassen to accept the coefficients for the linear combination.
    # Or, simpler: 
    # The 2x2 Strassen decomposition is linear.
    # If we want to compute (A1 + A2) * (B1 + B2), we can't just do 4 separate products.
    # We must use the Strassen algorithm on the combined matrices.
    # 
    # So, for M1:
    # Local A matrix is A_sum. Its entries are linear combos of global A entries.
    # Local B matrix is B_sum.
    # 
    # We need to modify add_strassen to handle linear combinations of blocks.
    
    def add_strassen_comb(col_offset, a_coeffs, b_coeffs, c_coeffs):
        """
        a_coeffs: dict mapping local index (0-3) to list of (global_idx, coeff)
        """
        for local_i in range(4):
            for r in range(7):
                val_u = U_strassen[local_i][r]
                val_v = V_strassen[local_i][r]
                val_w = W_strassen[local_i][r]
                
                if val_u != 0:
                    for g_idx, coeff in a_coeffs[local_i]:
                        U[g_idx][col_offset + r] += val_u * coeff
                if val_v != 0:
                    for g_idx, coeff in b_coeffs[local_i]:
                        V[g_idx][col_offset + r] += val_v * coeff
                if val_w != 0:
                    for g_idx, coeff in c_coeffs[local_i]:
                        W[g_idx][col_offset + r] += val_w * coeff

    # Reset col
    col = 0
    
    # Helper to create coeffs for a single block
    def single_block_coeffs(indices):
        return {i: [(indices[i], 1)] for i in range(4)}
    
    # Helper to create coeffs for sum of two blocks
    def sum_block_coeffs(idx1, idx2):
        return {i: [(idx1[i], 1), (idx2[i], 1)] for i in range(4)}
    
    # Helper to create coeffs for diff of two blocks
    def diff_block_coeffs(idx1, idx2):
        return {i: [(idx1[i], 1), (idx2[i], -1)] for i in range(4)}
    
    # M1 = (A00+A11)(B00+B11)
    add_strassen_comb(col, sum_block_coeffs(a00, a11), sum_block_coeffs(b00, b11), single_block_coeffs(c00)); col += 7
    # M2 = (A10+A11)B00
    add_strassen_comb(col, sum_block_coeffs(a10, a11), single_block_coeffs(b00), single_block_coeffs(c00)); col += 7
    # M3 = A00(B01-B11)
    add_strassen_comb(col, single_block_coeffs(a00), diff_block_coeffs(b01, b11), single_block_coeffs(c01)); col += 7
    # M4 = A11(B10-B00)
    add_strassen_comb(col, single_block_coeffs(a11), diff_block_coeffs(b10, b00), single_block_coeffs(c00)); col += 7
    # M5 = (A00+A10)B11
    add_strassen_comb(col, sum_block_coeffs(a00, a10), single_block_coeffs(b11), single_block_coeffs(c01)); col += 7
    # M6 = (A11-A00)(B00+B01)
    add_strassen_comb(col, diff_block_coeffs(a11, a00), sum_block_coeffs(b00, b01), single_block_coeffs(c11)); col += 7
    # M7 = (A01-A11)(B10+B11)
    add_strassen_comb(col, diff_block_coeffs(a01, a11), sum_block_coeffs(b10, b11), single_block_coeffs(c11)); col += 7
    
    # The W coefficients above are just single_block_coeffs(cXX).
    # But the Strassen W matrix combines the M_i's.
    # My add_strassen_comb adds the W contribution for a single M_i.
    # The final C is a sum of M_i's with specific coefficients.
    # The W_strassen matrix already contains these coefficients!
    # W_strassen[local_c, r] is the coefficient of M_r for C_local_c.
    # So, by adding W_strassen * (coeffs for C block), we are correctly accumulating the contributions.
    # 
    # Wait, in add_strassen_comb, I use `val_w` from W_strassen.
    # And I multiply it by `coeff` from c_coeffs.
    # This is correct.
    
    # --- Part 2: 4x4x2 (Rank 28) ---
    # A is 4x4, B is 4x2 (cols 4-5 of B), C is 4x2 (cols 4-5 of C)
    
    # B_top: j in 0,1; k in 4,5
    b_top = [4, 5, 10, 11]
    # B_bottom: j in 2,3; k in 4,5
    b_bot = [16, 17, 22, 23]
    
    # C_top: k in 4,5; i in 0,1
    c_top = [16, 17, 20, 21]
    # C_bottom: k in 4,5; i in 2,3
    c_bot = [18, 19, 22, 23]
    
    # 1. A00 * B_top -> C_top
    add_strassen_comb(col, single_block_coeffs(a00), single_block_coeffs(b_top), single_block_coeffs(c_top)); col += 7
    # 2. A01 * B_bottom -> C_top
    add_strassen_comb(col, single_block_coeffs(a01), single_block_coeffs(b_bot), single_block_coeffs(c_top)); col += 7
    # 3. A10 * B_top -> C_bottom
    add_strassen_comb(col, single_block_coeffs(a10), single_block_coeffs(b_top), single_block_coeffs(c_bot)); col += 7
    # 4. A11 * B_bottom -> C_bottom
    add_strassen_comb(col, single_block_coeffs(a11), single_block_coeffs(b_bot), single_block_coeffs(c_bot)); col += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
