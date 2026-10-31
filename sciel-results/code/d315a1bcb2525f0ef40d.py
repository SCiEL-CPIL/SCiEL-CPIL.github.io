"""Rewritten program for 4x4 matrix multiplication using Strassen's algorithm (Rank 49)."""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49  # 7 (top-level) * 7 (bottom-level)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get row index for U (A matrix, n x m)
    # A is 4x4. Blocks are 2x2.
    # Block (br, bc) corresponds to rows 2*br..2*br+1, cols 2*bc..2*bc+1.
    # Element (i, j) in A:
    # br = i // 2, bc = j // 2
    # local row ir = i % 2, local col ic = j % 2
    
    # Strassen 2x2 definitions for coefficients of A, B, C
    # Each entry is a dict: { (block_row, block_col): coeff }
    # Or we can just list the linear combinations.
    
    # Top level Strassen (for 4x4 blocks, treating each 2x2 block as a scalar)
    # A_blocks: A00, A01, A10, A11
    # B_blocks: B00, B10, B01, B11 (Note: B is m x p, so B is 4x4. Blocks B00 (top-left), B01 (top-right), B10 (bottom-left), B11 (bottom-right))
    # C_blocks: C00, C01, C10, C11
    
    # Define the 7 top-level multiplications M1..M7
    # M1 = (A11 + A22)(B11 + B22) -> In 0-indexed blocks: (A11+A00)? No.
    # Standard 1-indexed: A11 is top-left.
    # Let's use 0-indexed blocks: A00 (top-left), A01 (top-right), A10 (bottom-left), A11 (bottom-right).
    # Strassen formulas (1-indexed in literature, mapping to 0-indexed):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each top-level term t (1..7), we have a linear combination of A blocks and B blocks.
    # Let's represent the A-part and B-part of each Mt as a list of (block_idx, coeff).
    # block_idx 0: A00, 1: A01, 2: A10, 3: A11
    # block_idx 0: B00, 1: B01, 2: B10, 3: B11
    
    top_A = [
        [(0, 1), (3, 1)],      # M1: A00 + A11
        [(2, 1), (3, 1)],      # M2: A10 + A11
        [(0, 1)],              # M3: A00
        [(3, 1)],              # M4: A11
        [(0, 1), (1, 1)],      # M5: A00 + A01
        [(2, 1), (0, -1)],     # M6: A10 - A00
        [(1, 1), (3, -1)]      # M7: A01 - A11
    ]
    
    top_B = [
        [(0, 1), (3, 1)],      # M1: B00 + B11
        [(0, 1)],              # M2: B00
        [(1, 1), (3, -1)],     # M3: B01 - B11
        [(2, 1), (0, -1)],     # M4: B10 - B00
        [(3, 1)],              # M5: B11
        [(0, 1), (1, 1)],      # M6: B00 + B01
        [(2, 1), (3, 1)]       # M7: B10 + B11
    ]
    
    # Bottom level Strassen (for 2x2 blocks, treating each scalar as a scalar)
    # The blocks in top level are 2x2 matrices. Let's call them X and Y.
    # We multiply X * Y using Strassen.
    # X00, X01, X10, X11 are the scalar elements of the 2x2 block X.
    # Y00, Y01, Y10, Y11 are the scalar elements of the 2x2 block Y.
    
    # The 7 bottom-level multiplications m1..m7:
    # m1 = (X00 + X11) * (Y00 + Y11)
    # m2 = (X10 + X11) * Y00
    # m3 = X00 * (Y01 - Y11)
    # m4 = X11 * (Y10 - Y00)
    # m5 = (X00 + X01) * Y11
    # m6 = (X10 - X00) * (Y00 + Y01)
    # m7 = (X01 - X11) * (Y10 + Y11)
    
    # Output of bottom level (2x2 result Z):
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    
    # We need to map the scalar elements of X (which are elements of A) and Y (elements of B) to the global indices.
    
    # Let's iterate over top-level term t and bottom-level term s.
    # For a fixed t, X is a linear combo of A blocks, Y is a linear combo of B blocks.
    # For a fixed s, we pick specific scalar elements from X and Y.
    
    # Let's pre-compute the "bottom" coefficients for A and B scalars.
    # For bottom term s, we need to know which scalars of X and Y are involved and with what coefficients.
    
    # X scalars: X00, X01, X10, X11.
    # Y scalars: Y00, Y01, Y10, Y11.
    
    bottom_A_coeffs = [
        {(0,0): 1, (1,1): 1},      # m1: X00 + X11
        {(1,0): 1, (1,1): 1},      # m2: X10 + X11
        {(0,0): 1},                # m3: X00
        {(1,1): 1},                # m4: X11
        {(0,0): 1, (0,1): 1},      # m5: X00 + X01
        {(1,0): 1, (0,0): -1},     # m6: X10 - X00
        {(0,1): 1, (1,1): -1}      # m7: X01 - X11
    ]
    
    bottom_B_coeffs = [
        {(0,0): 1, (1,1): 1},      # m1: Y00 + Y11
        {(0,0): 1},                # m2: Y00
        {(0,1): 1, (1,1): -1},     # m3: Y01 - Y11
        {(1,0): 1, (0,0): -1},     # m4: Y10 - Y00
        {(1,1): 1},                # m5: Y11
        {(0,0): 1, (0,1): 1},      # m6: Y00 + Y01
        {(1,0): 1, (1,1): 1}       # m7: Y10 + Y11
    ]
    
    # Now, for each t (0..6) and s (0..6):
    # Column index c = t * 7 + s
    
    for t in range(7):
        for s in range(7):
            c = t * 7 + s
            
            # 1. Determine A coefficients
            # X is sum over top_A[t] blocks.
            # X_scalar(local_r, local_c) = sum over (block_idx, coeff) in top_A[t] of coeff * A_element(block_r, block_c, local_r, local_c)
            # A_element(block_r, block_c, local_r, local_c) corresponds to global row i = 2*block_r + local_r, col j = 2*block_c + local_c.
            
            # We need U[i*m+j][c] for all i,j.
            # Initialize U_col as zeros, then add contributions.
            U_col = [0] * (n * m)
            
            for (block_idx_a, coeff_a) in top_A[t]:
                br_a = block_idx_a // 2
                bc_a = block_idx_a % 2
                for (local_r, local_c), coeff_bottom_a in bottom_A_coeffs[s].items():
                    i = 2 * br_a + local_r
                    j = 2 * bc_a + local_c
                    idx_u = i * m + j
                    U_col[idx_u] += coeff_a * coeff_bottom_a
            
            for idx, val in enumerate(U_col):
                U[idx][c] = val
            
            # 2. Determine B coefficients
            # Y is sum over top_B[t] blocks.
            # Y_scalar(local_r, local_c) = sum over (block_idx, coeff) in top_B[t] of coeff * B_element(block_r, block_c, local_r, local_c)
            # B_element(block_r, block_c, local_r, local_c) corresponds to global row j = 2*block_r + local_r, col k = 2*block_c + local_c.
            
            V_col = [0] * (m * p)
            
            for (block_idx_b, coeff_b) in top_B[t]:
                br_b = block_idx_b // 2
                bc_b = block_idx_b % 2
                for (local_r, local_c), coeff_bottom_b in bottom_B_coeffs[s].items():
                    j = 2 * br_b + local_r
                    k = 2 * bc_b + local_c
                    idx_v = j * p + k
                    V_col[idx_v] += coeff_b * coeff_bottom_b
            
            for idx, val in enumerate(V_col):
                V[idx][c] = val
            
            # 3. Determine C coefficients
            # The result of X*Y is a 2x2 matrix Z.
            # Z is added to C blocks according to top-level reconstruction.
            # Top-level reconstruction for C blocks:
            # C00 += M1 + M4 - M5 + M7
            # C01 += M3 + M5
            # C10 += M2 + M4
            # C11 += M1 - M2 + M3 + M6
            
            # We need to know which C block receives the contribution of Mt.
            # Let's define top_C_coeffs[t] as a dict {block_idx: coeff}
            
            top_C = [
                {(0, 1), (3, 1), (2, -1), (1, 1)}, # Wait, indices:
                # M1: C00(1), C11(1)
                # M2: C10(1), C11(-1)
                # M3: C01(1), C11(1)
                # M4: C00(1), C10(1)
                # M5: C00(-1), C01(1)
                # M6: C11(1)
                # M7: C00(1)
                
                # Let's re-verify:
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                
                # So for M1 (t=0): C00:1, C11:1
                # M2 (t=1): C10:1, C11:-1
                # M3 (t=2): C01:1, C11:1
                # M4 (t=3): C00:1, C10:1
                # M5 (t=4): C00:-1, C01:1
                # M6 (t=5): C11:1
                # M7 (t=6): C00:1
            ]
            
            # Correct top_C list:
            top_C = [
                [(0, 1), (3, 1)],      # M1
                [(2, 1), (3, -1)],     # M2
                [(1, 1), (3, 1)],      # M3
                [(0, 1), (2, 1)],      # M4
                [(0, -1), (1, 1)],     # M5
                [(3, 1)],              # M6
                [(0, 1)]               # M7
            ]
            
            # Bottom-level reconstruction for Z (2x2 result of X*Y):
            # Z00 = m1 + m4 - m5 + m7
            # Z01 = m3 + m5
            # Z10 = m2 + m4
            # Z11 = m1 - m2 + m3 + m6
            
            # We need bottom_C_coeffs[s] as dict {(local_r, local_c): coeff}
            bottom_C = [
                {(0,0): 1, (1,0): 1, (1,1): -1, (1,1): 1}, # Error in manual dict creation, use lists
            ]
            
            # Let's define bottom_C properly
            bottom_C = [
                [(0,0, 1), (1,0, 1), (1,1, -1), (1,1, 1)], # No, Z00 is (0,0), Z10 is (1,0), etc.
                # m1 (s=0): Z00:1, Z11:1
                # m2 (s=1): Z10:1, Z11:-1
                # m3 (s=2): Z01:1, Z11:1
                # m4 (s=3): Z00:1, Z10:1
                # m5 (s=4): Z00:-1, Z01:1
                # m6 (s=5): Z11:1
                # m7 (s=6): Z00:1
            ]
            
            bottom_C = [
                [(0,0, 1), (1,1, 1)],      # m1
                [(1,0, 1), (1,1, -1)],     # m2
                [(0,1, 1), (1,1, 1)],      # m3
                [(0,0, 1), (1,0, 1)],      # m4
                [(0,0, -1), (0,1, 1)],     # m5
                [(1,1, 1)],                # m6
                [(0,0, 1)]                 # m7
            ]
            
            W_col = [0] * (p * n)
            
            # For each C block that Mt contributes to:
            for (block_idx_c, coeff_top_c) in top_C[t]:
                br_c = block_idx_c // 2
                bc_c = block_idx_c % 2
                # For each local position in the 2x2 block Z that ms contributes to:
                for (local_r, local_c, coeff_bottom_c) in bottom_C[s]:
                    # Global row i, col k
                    i = 2 * br_c + local_r
                    k = 2 * bc_c + local_c
                    # W row index: k * n + i
                    idx_w = k * n + i
                    W_col[idx_w] += coeff_top_c * coeff_bottom_c
            
            for idx, val in enumerate(W_col):
                W[idx][c] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
