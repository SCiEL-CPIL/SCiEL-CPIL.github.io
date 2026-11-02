"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication
    using a block-Strassen approach.
    
    Partition each 6x6 matrix into 3x3 blocks of 2x2 submatrices.
    Apply Strassen's rank-7 decomposition to the 3x3 block multiplication.
    This reduces the total rank from 216 (schoolbook) to 9 * 7 = 63 for the 
    block-level multiplications, but we need to account for the block structure.
    
    Actually, let me think more carefully. The tensor for 6x6 matrix multiplication
    can be decomposed by considering it as 3x3 block matrices where each block is 2x2.
    
    Let A, B, C be 6x6 matrices. Partition into 3x3 blocks of 2x2 matrices:
    A = [A_ij], B = [B_jk], C = [C_ik] where i,j,k in {0,1,2} and each A_ij etc is 2x2.
    
    C_ik = sum_j A_ij * B_jk (standard block matrix multiplication)
    
    The total rank is the sum over all (i,k) pairs of the rank of the tensor for
    computing C_ik from the relevant blocks. But this isn't quite right either.
    
    Let me think differently. The full tensor has n*m*p = 216 terms in the schoolbook.
    Using block structure: we have 3x3 = 9 block positions for the result, and for each
    result block C_ik, we need to compute sum_j A_ij * B_jk.
    
    For each fixed (i,k), the computation involves 3 terms (j=0,1,2), each being a 2x2
    matrix product. A 2x2 matrix product can be done with rank 7 (Strassen) instead of rank 8.
    
    So for each (i,k) pair, instead of 3*8 = 24 rank-1 tensors, we use 3*7 = 21.
    Total: 9 * 21 = 189.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS
    
    # Block dimensions: 3x3 blocks of 2x2 submatrices
    nb = 3  # number of blocks per dimension
    bs = 2  # block size
    
    # Strassen's 7 intermediate products for 2x2 matrix multiplication
    # For X, Y (2x2 matrices), C = X*Y:
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y01 - Y11)
    # M4 = X11 * (Y10 - Y00)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block (i,k) in the result, and each j (summing index),
    # we need to compute A_ij * B_jk using Strassen.
    
    # The overall rank will be nb*nb * (nb * 7) = 9 * 21 = 189
    
    rank = nb * nb * (nb * 7)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i_block in range(nb):
        for k_block in range(nb):
            for j_block in range(nb):
                # For this (i_block, k_block, j_block), we apply Strassen to
                # compute A[i_block, j_block] * B[j_block, k_block]
                # and add its contribution to C[i_block, k_block].
                
                # Strassen's 7 terms
                for s in range(7):
                    # Determine which elements of A_block and B_block are used
                    # and what coefficients they have, and how they contribute to C_block.
                    
                    # A_block = A[i_block*bs : (i_block+1)*bs, j_block*bs : (j_block+1)*bs]
                    # B_block = B[j_block*bs : (j_block+1)*bs, k_block*bs : (k_block+1)*bs]
                    # C_block += A_block * B_block (via Strassen)
                    
                    # Let's define the Strassen intermediates more carefully.
                    # A_block elements: A00, A01, A10, A11 (row-major within block)
                    # B_block elements: B00, B01, B10, B11
                    
                    # For each of the 7 Strassen products, we have:
                    # Product = sum of (coeff * A_elem) * sum of (coeff * B_elem)
                    # which expands to a rank-1 tensor (outer product of sums)
                    
                    # Actually, each Strassen intermediate is a product of two linear
                    # combinations, so each contributes rank 1 to the tensor.
                    
                    # Let me define for each s in 0..6:
                    # X_s = linear combination of A_block elements
                    # Y_s = linear combination of B_block elements
                    # Then M_s = X_s * Y_s
                    # And C_block = sum over s of (linear combination of M_s)
                    
                    # The contribution to the full tensor from this (i_block, k_block, j_block, s)
                    # is: for each a_idx in X_s's support and b_idx in Y_s's support,
                    # U[row_a, col] += coeff_a, V[col_b, col] += coeff_b
                    # And W gets the contribution from M_s to C_block elements.
                    
                    # Let me define the Strassen factors:
                    # s=0: X = A00 + A11, Y = B00 + B11
                    # s=1: X = A10 + A11, Y = B00
                    # s=2: X = A00, Y = B01 - B11
                    # s=3: X = A11, Y = B10 - B00
                    # s=4: X = A00 + A01, Y = B11
                    # s=5: X = A10 - A00, Y = B00 + B01
                    # s=6: X = A01 - A11, Y = B10 + B11
                    
                    # C00 = M0 + M3 - M4 + M6
                    # C01 = M2 + M4
                    # C10 = M1 + M3
                    # C11 = M0 - M1 + M2 + M5
                    
                    # Local indices within 2x2 block: (0,0), (0,1), (1,0), (1,1)
                    # A_block local index: (lr, lc) where lr,lc in {0,1}
                    # Global A index: row = i_block*bs + lr, col = j_block*bs + lc
                    # B_block local index: (lr, lc)
                    # Global B index: row = j_block*bs + lr, col = k_block*bs + lc
                    # C_block local index: (lr, lc)
                    # Global C index: row = i_block*bs + lr, col = k_block*bs + lc
                    
                    # For each Strassen term s, define the local coefficients:
                    if s == 0:
                        # X = A00 + A11, Y = B00 + B11
                        x_coeffs = {((0,0)): 1, ((1,1)): 1}
                        y_coeffs = {((0,0)): 1, ((1,1)): 1}
                        # Contribution to C: C00 += M0, C11 += M0
                        c_contrib = {((0,0)): 1, ((1,1)): 1}
                    elif s == 1:
                        # X = A10 + A11, Y = B00
                        x_coeffs = {((1,0)): 1, ((1,1)): 1}
                        y_coeffs = {((0,0)): 1}
                        # C10 += M1, C11 -= M1
                        c_contrib = {((1,0)): 1, ((1,1)): -1}
                    elif s == 2:
                        # X = A00, Y = B01 - B11
                        x_coeffs = {((0,0)): 1}
                        y_coeffs = {((0,1)): 1, ((1,1)): -1}
                        # C01 += M2, C11 += M2
                        c_contrib = {((0,1)): 1, ((1,1)): 1}
                    elif s == 3:
                        # X = A11, Y = B10 - B00
                        x_coeffs = {((1,1)): 1}
                        y_coeffs = {((1,0)): 1, ((0,0)): -1}
                        # C00 += M3, C10 += M3
                        c_contrib = {((0,0)): 1, ((1,0)): 1}
                    elif s == 4:
                        # X = A00 + A01, Y = B11
                        x_coeffs = {((0,0)): 1, ((0,1)): 1}
                        y_coeffs = {((1,1)): 1}
                        # C00 -= M4, C01 += M4
                        c_contrib = {((0,0)): -1, ((0,1)): 1}
                    elif s == 5:
                        # X = A10 - A00, Y = B00 + B01
                        x_coeffs = {((1,0)): 1, ((0,0)): -1}
                        y_coeffs = {((0,0)): 1, ((0,1)): 1}
                        # C11 += M5
                        c_contrib = {((1,1)): 1}
                    elif s == 6:
                        # X = A01 - A11, Y = B10 + B11
                        x_coeffs = {((0,1)): 1, ((1,1)): -1}
                        y_coeffs = {((1,0)): 1, ((1,1)): 1}
                        # C00 += M6
                        c_contrib = {((0,0)): 1}
                    
                    # Now add contributions to U, V, W
                    for (lr_a, lc_a), coeff_a in x_coeffs.items():
                        global_a_row = i_block * bs + lr_a
                        global_a_col = j_block * bs + lc_a
                        U[global_a_row * m + global_a_col][col] += coeff_a
                    
                    for (lr_b, lc_b), coeff_b in y_coeffs.items():
                        global_b_row = j_block * bs + lr_b
                        global_b_col = k_block * bs + lc_b
                        V[global_b_row * p + global_b_col][col] += coeff_b
                    
                    for (lr_c, lc_c), coeff_c in c_contrib.items():
                        global_c_row = i_block * bs + lr_c
                        global_c_col = k_block * bs + lc_c
                        # W index: W[k*n+i, r] per the task convention
                        # The output factor follows W[k*n+i, r]
                        # So W index = k_block*bs + lc_c, i_block*bs + lr_c
                        w_row = global_c_col * n + global_c_row
                        W[w_row][col] += coeff_c
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
