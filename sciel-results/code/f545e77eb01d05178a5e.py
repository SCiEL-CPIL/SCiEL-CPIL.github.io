import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W lists
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        U.append(u_vals)
        V.append(v_vals)
        W.append(w_vals)
        
    # 1. Strassen decomposition for 4x4x4 (first 4 columns of B)
    # Blocks:
    # A11 (0,0)-(1,1), A12 (0,2)-(1,3)
    # A21 (2,0)-(3,1), A22 (2,2)-(3,3)
    # Similarly for B.
    
    # Strassen Products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Each M is a 2x2 matrix product.
    # We decompose each 2x2 product using schoolbook (4 terms).
    
    # Define the 7 combinations of A and B blocks
    # Each block is a 2x2 matrix.
    # We need to define the coefficients for A and B in each M.
    
    # M1: A = A11 + A22, B = B11 + B22
    # M2: A = A21 + A22, B = B11
    # M3: A = A11, B = B12 - B22
    # M4: A = A22, B = B21 - B11
    # M5: A = A11 + A12, B = B22
    # M6: A = A21 - A11, B = B11 + B12
    # M7: A = A12 - A22, B = B21 + B22
    
    # For each M, we generate 4 terms for the 2x2 multiplication.
    # 2x2 schoolbook:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # We need to map the local 2x2 indices (0,1) to global 4x4 indices.
    # Block A11: global rows 0,1; cols 0,1
    # Block A12: global rows 0,1; cols 2,3
    # Block A21: global rows 2,3; cols 0,1
    # Block A22: global rows 2,3; cols 2,3
    
    # Let's define the block offsets
    # A_blocks:
    # 0: (0,0) -> rows 0,1, cols 0,1
    # 1: (0,2) -> rows 0,1, cols 2,3
    # 2: (2,0) -> rows 2,3, cols 0,1
    # 3: (2,2) -> rows 2,3, cols 2,3
    
    # B_blocks:
    # 0: (0,0) -> rows 0,1, cols 0,1 (for B11)
    # 1: (0,2) -> rows 0,1, cols 2,3 (for B12)
    # 2: (2,0) -> rows 2,3, cols 0,1 (for B21)
    # 3: (2,2) -> rows 2,3, cols 2,3 (for B22)
    
    # Note: B is 4x4 here.
    
    # Function to create a 2x2 term for a specific block multiplication
    # Inputs:
    # a_blocks: list of (block_id, coeff)
    # b_blocks: list of (block_id, coeff)
    # local_i, local_j: indices in the 2x2 result (0,1)
    # local_a_i, local_a_j: indices in the 2x2 A block
    # local_b_i, local_b_j: indices in the 2x2 B block
    
    # This is getting complicated to generalize.
    # Let's just hardcode the 28 terms for the 4x4x4 part.
    
    # Term generation for one 2x2 multiplication (schoolbook)
    # Result C is 2x2.
    # C[li, lj] = sum( A[li, la_j] * B[la_j, lj] )? No.
    # C[li, lj] = sum_k A[li, k] B[k, lj]
    
    # We will iterate through the 7 Strassen products.
    # For each product, we have a linear combination of A blocks and B blocks.
    # We expand this into 4 terms.
    
    # Define the 7 Strassen steps as (A_coeffs, B_coeffs)
    # A_coeffs: dict {block_id: coeff}
    # B_coeffs: dict {block_id: coeff}
    
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    # Block offsets for 4x4 matrix
    # Block 0: rows 0-1, cols 0-1
    # Block 1: rows 0-1, cols 2-3
    # Block 2: rows 2-3, cols 0-1
    # Block 3: rows 2-3, cols 2-3
    
    def get_block_offset(block_id, dim=2):
        r = (block_id // 2) * dim
        c = (block_id % 2) * dim
        return r, c
    
    # For each step, generate 4 terms (for 2x2 schoolbook)
    for a_coeffs, b_coeffs in steps:
        # For each of the 4 scalar multiplications in 2x2 schoolbook
        # C[li, lj] += A[li, lk] * B[lk, lj]
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    # This term contributes to C[li, lj]
                    # It involves A[li, lk] and B[lk, lj]
                    
                    # Construct U, V, W vectors for this term
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    
                    # U part: A
                    # A is 4x4.
                    # The term involves A[li, lk] from the specific blocks.
                    # Wait, the Strassen step combines blocks.
                    # A_total[li, lk] = sum_{b_a} coeff_a[b_a] * A_block[b_a][li, lk]
                    # We need to add these to U.
                    
                    for b_a, coeff_a in a_coeffs.items():
                        r_off, c_off = get_block_offset(b_a)
                        # Global row: r_off + li
                        # Global col: c_off + lk
                        gr = r_off + li
                        gc = c_off + lk
                        u[gr * m + gc] += coeff_a
                        
                    # V part: B
                    # B is 4x4 (first 4 cols).
                    # B_total[lk, lj] = sum_{b_b} coeff_b[b_b] * B_block[b_b][lk, lj]
                    for b_b, coeff_b in b_coeffs.items():
                        r_off, c_off = get_block_offset(b_b)
                        # Global row: r_off + lk
                        # Global col: c_off + lj
                        gr = r_off + lk
                        gc = c_off + lj
                        v[gr * p + gc] += coeff_b
                        
                    # W part: C
                    # C is 4x4.
                    # The result of this 2x2 multiplication is added to C.
                    # The local result C[li, lj] corresponds to which global C?
                    # This is tricky. The Strassen products M1..M7 are 2x2 matrices.
                    # They are added to the final 4x4 C with specific coefficients.
                    # M1 contributes to C11, C12, C21, C22?
                    # C11 = M11 + M41 - M51 + M71
                    # C12 = M12 + M52
                    # C21 = M21 + M42
                    # C22 = M12 - M22 + M32 + M62 ? No.
                    
                    # Standard Strassen reconstruction:
                    # C11 = M11 + M41 - M51 + M71
                    # C12 = M31 + M52  (Wait, M3 is 2x2? Yes.)
                    # C21 = M21 + M42
                    # C22 = M12 - M22 + M32 + M62
                    
                    # So we need to know which M (step index) and which local position (li, lj) we are at.
                    # And then apply the reconstruction coefficients to W.
                    
                    step_idx = steps.index((a_coeffs, b_coeffs))
                    
                    # Reconstruction coefficients for W
                    # W index: (col, row) in p*n?
                    # W[k*n+i, r]. k is col of C, i is row of C.
                    # Global row of C: r_off_c + li
                    # Global col of C: c_off_c + lj
                    # Which block of C is this?
                    # The Strassen products are for the 4x4 matrix.
                    # M1 is top-left? No, M1 is a 2x2 matrix.
                    # M1 contributes to all 4 blocks of C?
                    # No, M1 is the product of (A11+A22) and (B11+B22).
                    # This is a 2x2 matrix.
                    # Where does it go?
                    # It contributes to C11, C12, C21, C22?
                    # Actually, the Strassen algorithm computes 7 2x2 matrices.
                    # Then combines them to form the 4x4 C.
                    
                    # Let's check the mapping.
                    # C11 (top-left 2x2) = M11 + M41 - M51 + M71
                    # C12 (top-right 2x2) = M31 + M52  (Indices: M3[0,0] + M5[0,1]?)
                    # This is getting confusing without a reference.
                    
                    # Alternative: Just use the schoolbook for 4x4x4? Rank 64.
                    # Plus 16 for 5th col? Rank 80. No improvement.
                    
                    # Let's stick to the Block Strassen but verify the reconstruction.
                    # I will assume the standard Strassen reconstruction.
                    # M1: (A11+A22)(B11+B22)
                    # M2: (A21+A22)B11
                    # M3: A11(B12-B22)
                    # M4: A22(B21-B11)
                    # M5: (A11+A12)B22
                    # M6: (A21-A11)(B11+B12)
                    # M7: (A12-A22)(B21+B22)
                    
                    # C11 = M1 + M4 - M5 + M7 (Top-Left block)
                    # C12 = M3 + M5 (Top-Right block)
                    # C21 = M2 + M4 (Bottom-Left block)
                    # C22 = M1 - M2 + M3 + M6 (Bottom-Right block)
                    
                    # So, for a given step_idx (0..6) and local (li, lj):
                    # If step_idx == 0 (M1):
                    #   Contributes to C11[li,lj] with +1
                    #   Contributes to C22[li,lj] with +1
                    # If step_idx == 1 (M2):
                    #   Contributes to C21[li,lj] with +1
                    #   Contributes to C22[li,lj] with -1
                    # If step_idx == 2 (M3):
                    #   Contributes to C12[li,lj] with +1
                    #   Contributes to C22[li,lj] with +1
                    # If step_idx == 3 (M4):
                    #   Contributes to C11[li,lj] with +1
                    #   Contributes to C21[li,lj] with +1
                    # If step_idx == 4 (M5):
                    #   Contributes to C11[li,lj] with -1
                    #   Contributes to C12[li,lj] with +1
                    # If step_idx == 5 (M6):
                    #   Contributes to C22[li,lj] with +1
                    # If step_idx == 6 (M7):
                    #   Contributes to C11[li,lj] with +1

                    # We need to map (step_idx, li, lj) to global C indices.
                    # C11 is block 0 (rows 0-1, cols 0-1)
                    # C12 is block 1 (rows 0-1, cols 2-3)
                    # C21 is block 2 (rows 2-3, cols 0-1)
                    # C22 is block 3 (rows 2-3, cols 2-3)
                    
                    w_coeff = 0
                    c_block = -1
                    
                    if step_idx == 0: # M1
                        w_coeff += 1; c_block = 0 # C11
                        w_coeff += 1; c_block = 3 # C22 (Wait, can't add two blocks in one term? No, W is a vector. We can have non-zero entries in multiple places.)
                        # Actually, W is a column vector. We can set multiple entries.
                        # So we just add the contributions.
                        pass # Handled below
                    
                    # Let's just add to W directly.
                    
                    def add_w_contrib(block_id, coeff):
                        r_off, c_off = get_block_offset(block_id)
                        gr = r_off + li
                        gc = c_off + lj
                        # W index: gc * n + gr
                        w[gc * n + gr] += coeff

                    if step_idx == 0: # M1
                        add_w_contrib(0, 1)
                        add_w_contrib(3, 1)
                    elif step_idx == 1: # M2
                        add_w_contrib(2, 1)
                        add_w_contrib(3, -1)
                    elif step_idx == 2: # M3
                        add_w_contrib(1, 1)
                        add_w_contrib(3, 1)
                    elif step_idx == 3: # M4
                        add_w_contrib(0, 1)
                        add_w_contrib(2, 1)
                    elif step_idx == 4: # M5
                        add_w_contrib(0, -1)
                        add_w_contrib(1, 1)
                    elif step_idx == 5: # M6
                        add_w_contrib(3, 1)
                    elif step_idx == 6: # M7
                        add_w_contrib(0, 1)

                    if any(x != 0 for x in u) and any(x != 0 for x in v) and any(x != 0 for x in w):
                        add_term(u, v, w)

    # 2. Add terms for the 5th column (index 4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Rank 4 terms.
    for j in range(4):
        u = [0] * (n*m)
        v = [0] * (m*p)
        w = [0] * (p*n)
        
        # A[i, j] for all i
        for i in range(4):
            u[i * m + j] = 1
            
        # B[j, 4]
        v[j * p + 4] = 1
        
        # C[i, 4] for all i
        for i in range(4):
            w[4 * n + i] = 1
            
        add_term(u, v, w)

    # Convert to lists of lists
    U_out = [row[:] for row in U]
    V_out = [row[:] for row in V]
    W_out = [row[:] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
