import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5
    # Rank 49 for 4x4x4, Rank 16 for 4x4x1
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Multiplication (Rank 49) ---
    # Strassen's algorithm for 2x2 matrices
    # Terms:
    # S1 = (X00+X11)(Y00+Y11)
    # S2 = (X10+X11)Y00
    # S3 = X00(Y01-Y11)
    # S4 = X11(Y10-Y00)
    # S5 = (X00+X01)Y11
    # S6 = (X10-X00)(Y00+Y01)
    # S7 = (X01-X11)(Y10+Y11)
    
    # Define the 7 linear forms for X (left) and Y (right)
    # X indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Y indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    x_forms = [
        [1, 0, 0, 1], # X00+X11
        [0, 0, 1, 1], # X10+X11
        [1, 0, 0, 0], # X00
        [0, 0, 0, 1], # X11
        [1, 1, 0, 0], # X00+X01
        [0, 0, 1, -1],# X10-X00
        [0, 1, 0, -1] # X01-X11
    ]
    
    y_forms = [
        [1, 0, 0, 1], # Y00+Y11
        [1, 0, 0, 0], # Y00
        [0, 1, 0, -1],# Y01-Y11
        [0, 0, 1, -1],# Y10-Y00
        [0, 0, 0, 1], # Y11
        [1, 1, 0, 0], # Y00+Y01
        [0, 0, 1, 1]  # Y10+Y11
    ]
    
    # Outer Strassen steps for 4x4
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M_k is a 2x2 matrix product.
    # We need to map the 2x2 blocks of A and B to the 4x4 indices.
    # A:
    # 00 01 10 11
    # A00 = A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,4,5
    # A01 = A[0,2], A[0,3], A[1,2], A[1,3] -> indices 2,3,6,7
    # A10 = A[2,0], A[2,1], A[3,0], A[3,1] -> indices 8,9,12,13
    # A11 = A[2,2], A[2,3], A[3,2], A[3,3] -> indices 10,11,14,15
    
    # B (4x4 part):
    # B00 = B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,4,5
    # B01 = B[0,2], B[0,3], B[1,2], B[1,3] -> indices 2,3,6,7
    # B10 = B[2,0], B[2,1], B[3,0], B[3,1] -> indices 8,9,12,13
    # B11 = B[2,2], B[2,3], B[3,2], B[3,3] -> indices 10,11,14,15
    
    # Helper to get 4 indices for a block
    def get_a_block_indices(block):
        if block == 0: return [0, 1, 4, 5]
        if block == 1: return [2, 3, 6, 7]
        if block == 2: return [8, 9, 12, 13]
        if block == 3: return [10, 11, 14, 15]
        
    def get_b_block_indices(block):
        # B is 4x4 for this part
        if block == 0: return [0, 1, 4, 5]
        if block == 1: return [2, 3, 6, 7]
        if block == 2: return [8, 9, 12, 13]
        if block == 3: return [10, 11, 14, 15]

    # Define the 7 outer steps: (left_block1, left_block2, right_block1, right_block2, coeffs)
    # L = c1 * A_b1 + c2 * A_b2
    # R = c3 * B_b3 + c4 * B_b4
    # Note: Strassen uses sums/diffs.
    
    outer_steps = [
        # (L_idx1, L_idx2, R_idx1, R_idx2, coeff_L1, coeff_L2, coeff_R1, coeff_R2)
        (0, 3, 0, 3, 1, 1, 1, 1),   # M1: (A00+A11)(B00+B11)
        (2, 3, 0, -1, 1, 1, 1, 0),  # M2: (A10+A11)B00  (R is just B00)
        (0, -1, 1, 3, 1, 0, 1, -1), # M3: A00(B01-B11)
        (3, -1, 2, 0, 1, 0, 1, -1), # M4: A11(B10-B00)
        (0, 1, 3, -1, 1, 1, 1, 0),  # M5: (A00+A01)B11
        (2, 0, 0, 1, 1, -1, 1, 1),  # M6: (A10-A00)(B00+B01)
        (1, 3, 2, 3, 1, -1, 1, 1)   # M7: (A01-A11)(B10+B11)
    ]
    
    col = 0
    for step_idx in range(7):
        l1, l2, r1, r2, cl1, cl2, cr1, cr2 = outer_steps[step_idx]
        
        # For each of the 7 inner Strassen terms
        for inner_idx in range(7):
            # Get the linear forms for X (left) and Y (right)
            # x_forms[inner_idx] is a 4-vector [c00, c01, c10, c11]
            xf = x_forms[inner_idx]
            yf = y_forms[inner_idx]
            
            # Construct U coefficients for this term
            # U is 16x1. We add to U[:, col]
            # The left matrix L is 2x2.
            # L = cl1 * A_l1 + cl2 * A_l2 (if l2 != -1)
            
            # Indices in A for the 2x2 block l1
            idx_l1 = get_a_block_indices(l1)
            idx_l2 = get_a_block_indices(l2) if l2 != -1 else []
            
            # xf = [x00, x01, x10, x11]
            # L00 = xf[0] * (cl1 * A_l1[00] + cl2 * A_l2[00])
            # L01 = xf[1] * (cl1 * A_l1[01] + cl2 * A_l2[01])
            # L10 = xf[2] * (cl1 * A_l1[10] + cl2 * A_l2[10])
            # L11 = xf[3] * (cl1 * A_l1[11] + cl2 * A_l2[11])
            
            for k in range(4):
                # k=0 -> L00, k=1 -> L01, k=2 -> L10, k=3 -> L11
                coeff_x = xf[k]
                if coeff_x != 0:
                    # Add to U at indices corresponding to L's entries
                    # L00 is at idx_l1[0]
                    if l1 != -1:
                        U[idx_l1[k]][col] += coeff_x * cl1
                    if l2 != -1:
                        U[idx_l2[k]][col] += coeff_x * cl2
                        
            # Construct V coefficients for this term
            # R = cr1 * B_r1 + cr2 * B_r2 (if r2 != -1)
            idx_r1 = get_b_block_indices(r1)
            idx_r2 = get_b_block_indices(r2) if r2 != -1 else []
            
            for k in range(4):
                coeff_y = yf[k]
                if coeff_y != 0:
                    if r1 != -1:
                        V[idx_r1[k]][col] += coeff_y * cr1
                    if r2 != -1:
                        V[idx_r2[k]][col] += coeff_y * cr2
                        
            # Construct W coefficients for this term
            # The result M_k contributes to C.
            # C = sum M_k * W_k
            # We need to determine which C entries are affected and by what coefficient.
            
            # The inner Strassen term S_inner contributes to the 2x2 result M_k.
            # Let's map the 2x2 result entries to the 4x4 C entries.
            
            # M_k is 2x2. Let's call its entries M00, M01, M10, M11.
            # The inner Strassen algorithm computes:
            # M00 = S1 + S4 - S5 + S7
            # M01 = S3 + S5
            # M10 = S2 + S4
            # M11 = S1 - S2 + S3 + S6
            
            # So for a specific inner_idx, we know which M entries it affects.
            # inner_idx 0 (S1): M00(+1), M11(+1)
            # inner_idx 1 (S2): M10(+1), M11(-1)
            # inner_idx 2 (S3): M01(+1), M11(+1)
            # inner_idx 3 (S4): M00(+1), M10(+1)
            # inner_idx 4 (S5): M00(-1), M01(+1)
            # inner_idx 5 (S6): M11(+1)
            # inner_idx 6 (S7): M00(+1)
            
            w_contribs = [
                [(0, 1), (1, 1)],   # S1: M00, M11
                [(1, 0), (1, 1)],   # S2: M10, M11
                [(0, 1), (1, 1)],   # S3: M01, M11
                [(0, 0), (1, 0)],   # S4: M00, M10
                [(0, 0), (0, 1)],   # S5: M00, M01
                [(1, 1)],           # S6: M11
                [(0, 0)]            # S7: M00
            ]
            
            # Coefficients for the contributions
            # S1: +1 to M00, +1 to M11
            # S2: +1 to M10, -1 to M11
            # S3: +1 to M01, +1 to M11
            # S4: +1 to M00, +1 to M10
            # S5: -1 to M00, +1 to M01
            # S6: +1 to M11
            # S7: +1 to M00
            
            coeffs = [
                [1, 1],
                [1, -1],
                [1, 1],
                [1, 1],
                [-1, 1],
                [1],
                [1]
            ]
            
            # Now, map M_k's entries to C's entries.
            # M1 (k=0): C00 += M1_00, C01 += M1_01, C10 += M1_10, C11 += M1_11
            # M2 (k=1): C10 += M2_10? No.
            # Let's check the outer Strassen formulas again.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So for step_idx (0-based k):
            # k=0 (M1): C00(+1), C11(+1)
            # k=1 (M2): C10(+1), C11(-1)
            # k=2 (M3): C01(+1), C11(+1)
            # k=3 (M4): C00(+1), C10(+1)
            # k=4 (M5): C00(-1), C01(+1)
            # k=5 (M6): C11(+1)
            # k=6 (M7): C00(+1)
            
            outer_w_contribs = [
                [(0, 0, 1), (1, 1, 1)],   # M1: C00(+1), C11(+1)
                [(1, 0, 1), (1, 1, -1)],  # M2: C10(+1), C11(-1)
                [(0, 1, 1), (1, 1, 1)],   # M3: C01(+1), C11(+1)
                [(0, 0, 1), (1, 0, 1)],   # M4: C00(+1), C10(+1)
                [(0, 0, -1), (0, 1, 1)],  # M5: C00(-1), C01(+1)
                [(1, 1, 1)],              # M6: C11(+1)
                [(0, 0, 1)]               # M7: C00(+1)
            ]
            
            # Combine inner and outer contributions
            # For each (m_row, m_col, coeff_inner) in w_contribs[inner_idx]:
            #    For each (c_row, c_col, coeff_outer) in outer_w_contribs[step_idx]:
            #       If m_row == c_row and m_col == c_col? No.
            #       The M_k is a 2x2 block.
            #       M1 corresponds to C00 (top-left 2x2 block of C).
            #       Wait, C is 4x4.
            #       C00 is the top-left 2x2 block.
            #       So M1_00 contributes to C_00_00, M1_01 to C_00_01, etc.
            
            # Let's refine the mapping.
            # C is 4x4.
            # C00 block is rows 0-1, cols 0-1.
            # C01 block is rows 0-1, cols 2-3.
            # C10 block is rows 2-3, cols 0-1.
            # C11 block is rows 2-3, cols 2-3.
            
            # M1 contributes to C00 and C11?
            # Formula: C00 = M1 + ...
            # Here C00 is the top-left 2x2 block of the final 4x4 matrix C.
            # So M1 (2x2) adds directly to C[0:2, 0:2].
            # And M1 also adds to C11?
            # Formula: C11 = M1 - M2 + M3 + M6.
            # So M1 adds to C[2:4, 2:4] as well.
            
            # So for a given step_idx and inner_idx:
            # We have contributions to specific M entries (2x2).
            # Each M entry (mr, mc) maps to specific C blocks.
            
            # Let's iterate through the inner contributions
            inner_contribs = w_contribs[inner_idx]
            inner_coeffs = coeffs[inner_idx]
            
            for idx_c in range(len(inner_contribs)):
                mr, mc = inner_contribs[idx_c]
                coeff_in = inner_coeffs[idx_c]
                
                # Now apply outer mapping
                outer_contribs = outer_w_contribs[step_idx]
                for idx_o in range(len(outer_contribs)):
                    c_block_row, c_block_col, coeff_out = outer_contribs[idx_o]
                    
                    # The M entry (mr, mc) is part of the 2x2 matrix M_k.
                    # M_k is added to the C block at (c_block_row, c_block_col).
                    # So C[c_block_row*2 + mr, c_block_col*2 + mc] += coeff_in * coeff_out * (Scalar Product)
                    
                    c_row = c_block_row * 2 + mr
                    c_col = c_block_col * 2 + mc
                    
                    # W index: k*n + i -> c_col * 4 + c_row
                    w_idx = c_col * 4 + c_row
                    W[w_idx][col] += coeff_in * coeff_out
                    
            col += 1

    # --- Part 2: 4x4x1 Multiplication (Rank 16) ---
    # C[:, 4] = A * b
    # C[i, 4] = sum_j A[i, j] * b[j]
    # 16 terms: A[i, j] * b[j]
    
    for i in range(4):
        for j in range(4):
            # U[i*4 + j] = 1
            U[i * 4 + j][col] = 1
            # V[j*5 + 4] = 1 (b is the 5th column, index 4)
            V[j * 5 + 4][col] = 1
            # W[4*4 + i] = 1 (C[i, 4] -> index 4*4 + i = 16 + i)
            W[4 * 4 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
