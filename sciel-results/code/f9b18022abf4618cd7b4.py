import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will construct rank 112 decomposition
    # 2 * (7 Strassen terms * 8 schoolbook terms for 2x2)
    
    rank = 112
    U = [[0] * rank for _ in range(n * m)] # 16 x 112
    V = [[0] * rank for _ in range(m * p)] # 32 x 112
    W = [[0] * rank for _ in range(p * n)] # 32 x 112
    
    # Helper to add a rank-1 component
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # coeff_u, coeff_v, coeff_w
    # Actually, the decomposition is sum of rank 1 tensors.
    # Tensor T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # For schoolbook 2x2x2:
    # C[k,i] = sum_j A[i,j] B[j,k]
    # Term (i,j,k): U[i,j]=1, V[j,k]=1, W[k,i]=1
    
    # We are building 4x4x4 using Strassen.
    # A is 4x4. Blocks A00, A01, A10, A11 (each 2x2).
    # B1 is 4x4 (first 4 cols of B). Blocks B00, B01, B10, B11.
    # C1 is 4x4 (first 4 cols of C). Blocks C00, C01, C10, C11.
    
    # Strassen Formulas for 4x4:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product.
    # We decompose each M using schoolbook 2x2x2 (8 terms).
    
    # Indices for 2x2 blocks:
    # Block 00: rows 0-1, cols 0-1
    # Block 01: rows 0-1, cols 2-3
    # Block 10: rows 2-3, cols 0-1
    # Block 11: rows 2-3, cols 2-3
    
    # Function to add a 2x2x2 schoolbook decomposition scaled and shifted
    # A_block_indices: list of (block_row, block_col, coeff) for the left matrix
    # B_block_indices: list of (block_row, block_col, coeff) for the right matrix
    # C_block_indices: list of (block_row, block_col, coeff) for the output matrix
    # offset: rank offset
    # b_offset: column offset in B (0 for first half, 4 for second half)
    # c_offset: column offset in C (0 for first half, 4 for second half)
    
    def add_2x2_product(a_blocks, b_blocks, c_blocks, offset, b_col_offset, c_col_offset):
        # a_blocks: list of (br, bc, coeff) where br, bc in {0,1}
        # b_blocks: list of (br, bc, coeff)
        # c_blocks: list of (br, bc, coeff)
        
        # For each of the 8 schoolbook terms (i, j, k) in 2x2:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    r = offset + (i * 4 + j) * 2 + k # 8 terms per M
                    # Wait, 8 terms per M. 
                    # Let's map (i,j,k) to a sub-index 0..7
                    sub_idx = i * 4 + j * 2 + k
                    
                    # U part:
                    # U index is global (I, J)
                    # I = br * 2 + i, J = bc * 2 + j
                    # For each a_block (br, bc, coeff):
                    for (br, bc, coeff) in a_blocks:
                        I = br * 2 + i
                        J = bc * 2 + j
                        U[I * m + J][r] += coeff
                        
                    # V part:
                    # V index is global (J, K)
                    # J = br * 2 + j, K = bc * 2 + k
                    # Note: J here is the column of A, which is row of B.
                    # In 2x2 product X Y = Z, X[i,j] Y[j,k] -> Z[k,i]
                    # So the 'j' in schoolbook term matches the column of X and row of Y.
                    for (br, bc, coeff) in b_blocks:
                        J = br * 2 + j
                        K = bc * 2 + k
                        # Global B index: row J, col K + b_col_offset
                        V[J * p + (K + b_col_offset)][r] += coeff
                        
                    # W part:
                    # W index is global (K, I)
                    # K = br * 2 + k, I = bc * 2 + i
                    for (br, bc, coeff) in c_blocks:
                        K = br * 2 + k
                        I = bc * 2 + i
                        # Global C index: row K + c_col_offset, col I
                        # W convention: W[k*n + i, r]
                        W[(K + c_col_offset) * n + I][r] += coeff

    # Define the 7 Strassen terms
    # Each term is (a_blocks, b_blocks, c_blocks)
    # a_blocks, b_blocks, c_blocks are lists of (br, bc, coeff)
    
    # M1 = (A00 + A11)(B00 + B11)
    a1 = [(0,0,1), (1,1,1)]
    b1 = [(0,0,1), (1,1,1)]
    c1 = [] # Output is not direct, M1 contributes to C00 and C11
    # We need to add M1 to C00 and C11 with appropriate signs.
    # C00 += M1, C11 += M1
    # So we call add_2x2_product twice? Or combine c_blocks?
    # add_2x2_product expects c_blocks to define the target C.
    # If we want M1 to go to C00 and C11, we can't do it in one call if the mapping is 1-to-1.
    # But M1 is a single 2x2 matrix. It contributes to C00 (top-left) and C11 (bottom-right)?
    # No, M1 is a 2x2 matrix.
    # C00 is a 2x2 matrix. C11 is a 2x2 matrix.
    # The formula is C00 = M1 + ... and C11 = M1 + ...
    # So the matrix M1 adds to C00 and also adds to C11.
    # This means the rank-1 tensors for M1 must be duplicated for the two target blocks?
    # Yes.
    
    # Let's refine add_2x2_product to take a list of target blocks.
    # targets: list of (c_br, c_bc, coeff)
    
    def add_strassen_term(a_blocks, b_blocks, targets, offset, b_col_offset, c_col_offset):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    r = offset + (i * 4 + j) * 2 + k
                    for (br, bc, coeff) in a_blocks:
                        I = br * 2 + i
                        J = bc * 2 + j
                        U[I * m + J][r] += coeff
                    for (br, bc, coeff) in b_blocks:
                        J = br * 2 + j
                        K = bc * 2 + k
                        V[J * p + (K + b_col_offset)][r] += coeff
                    for (c_br, c_bc, c_coeff) in targets:
                        K = c_br * 2 + k
                        I = c_bc * 2 + i
                        W[(K + c_col_offset) * n + I][r] += c_coeff

    # Strassen Terms for First Half (B cols 0-3, C cols 0-3)
    # M1
    add_strassen_term([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 0, 0, 0)
    # M2
    add_strassen_term([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)], 8, 0, 0)
    # M3
    add_strassen_term([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)], 16, 0, 0)
    # M4
    add_strassen_term([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], 24, 0, 0)
    # M5
    add_strassen_term([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)], 32, 0, 0)
    # M6
    add_strassen_term([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)], 40, 0, 0)
    # M7
    add_strassen_term([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 48, 0, 0)
    
    # Wait, I need to check the C block mappings carefully.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # My calls above:
    # M1: targets C00, C11. Correct.
    # M2: targets C10. Correct.
    # M3: targets C01, C11. 
    #   My call: targets [(0,1,1)]. Missing C11 contribution?
    #   Formula: C11 = ... + M3 ...
    #   So M3 should target C01 and C11.
    # M4: targets C00, C10. Correct.
    # M5: targets C00 (-1), C01 (+1). Correct.
    # M6: targets C11. Correct.
    # M7: targets C00, C11. Correct.
    
    # Let's re-verify M3 and M6 targets.
    # M3 = A00 (B01 - B11).
    # C01 = M3 + M5. So M3 -> C01.
    # C11 = M1 - M2 + M3 + M6. So M3 -> C11.
    # So M3 targets: C01, C11.
    
    # M6 = (A10 - A00) (B00 + B01).
    # C11 = ... + M6. So M6 -> C11.
    # Does M6 go anywhere else? No.
    
    # Let's re-run the first half with corrected targets.
    # Reset U,V,W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First Half (offset 0, b_col 0, c_col 0)
    # M1
    add_strassen_term([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 0, 0, 0)
    # M2
    add_strassen_term([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)], 8, 0, 0)
    # M3
    add_strassen_term([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)], 16, 0, 0)
    # M4
    add_strassen_term([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], 24, 0, 0)
    # M5
    add_strassen_term([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)], 32, 0, 0)
    # M6
    add_strassen_term([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)], 40, 0, 0)
    # M7
    add_strassen_term([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 48, 0, 0)
    
    # Second Half (offset 56, b_col 4, c_col 4)
    # Same structure, just shifted indices.
    # M1
    add_strassen_term([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 56, 4, 4)
    # M2
    add_strassen_term([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)], 64, 4, 4)
    # M3
    add_strassen_term([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)], 72, 4, 4)
    # M4
    add_strassen_term([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], 80, 4, 4)
    # M5
    add_strassen_term([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)], 88, 4, 4)
    # M6
    add_strassen_term([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)], 96, 4, 4)
    # M7
    add_strassen_term([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 104, 4, 4)
    
    return {"U": U, "V": V, "W": W}
