import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Use Strassen for 2x2 blocks of 3x3 matrices
    # Rank = 7 * (3*3*3) = 7 * 27 = 189
    
    # We need to construct U, V, W for 6x6x6
    # A is 6x6, viewed as 2x2 blocks of 3x3
    # B is 6x6, viewed as 2x2 blocks of 3x3
    # C is 6x6, viewed as 2x2 blocks of 3x3
    
    # Strassen for 2x2 blocks:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pi is a 3x3 matrix product.
    # We use schoolbook for 3x3, rank 27.
    
    # Total rank = 7 * 27 = 189.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a 3x3 schoolbook decomposition
    # block_a: (r, c) block of A (3x3)
    # block_b: (r, c) block of B (3x3)
    # block_c: (r, c) block of C (3x3)
    # sign: +1 or -1
    # offset: column offset in U, V, W
    
    def add_3x3_block(a_row, a_col, b_row, b_col, c_row, c_col, sign, offset):
        # A block is at rows 3*a_row to 3*a_row+2, cols 3*a_col to 3*a_col+2
        # B block is at rows 3*b_row to 3*b_row+2, cols 3*b_col to 3*b_col+2
        # C block is at rows 3*c_row to 3*c_row+2, cols 3*c_col to 3*c_col+2
        
        # Schoolbook for 3x3:
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Indices i, j, k in 0..2
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = offset + (i * 3 + j) * 3 + k
                    # U index: A[i, j] within block
                    u_idx = (a_row * 3 + i) * 3 + (a_col * 3 + j)
                    # V index: B[j, k] within block
                    v_idx = (b_row * 3 + j) * 3 + (b_col * 3 + k)
                    # W index: C[i, k] within block
                    # Prompt convention: W[k*n+i]
                    # Here C is 6x6.
                    # C block is at c_row, c_col.
                    # C index in 6x6: row = c_row*3 + i, col = c_col*3 + k
                    # Prompt index: col*6 + row?
                    # Seed: W[k*n+i] where i is row of A, k is col of B.
                    # In seed: C[k, i] = sum A[i, j] B[j, k].
                    # So C row is k, col is i.
                    # Here C[i, k] from schoolbook.
                    # So C row is i, col is k?
                    # Wait, the seed program computes C[k, i].
                    # My schoolbook computes C[i, k].
                    # So I need to map C[i, k] to W index corresponding to C[k, i]?
                    # No, the seed program's W index is k*n+i.
                    # If I compute C_std[i, k], and the prompt wants C_prompt[k, i] = C_std[i, k],
                    # then W index for C_std[i, k] should be (k*6 + i)?
                    # Let's check the seed again.
                    # Seed: U[i*m+j], V[j*p+k], W[k*n+i].
                    # This implies C[k, i] = sum A[i, j] B[j, k].
                    # So the output matrix is C where C[k, i] is the entry.
                    # This is the transpose of the standard C = A B.
                    # So if I use standard Strassen which computes C_std = A B,
                    # then C_std[i, k] corresponds to C_prompt[k, i].
                    # So W index for C_std[i, k] is k*6 + i.
                    
                    w_idx = (c_col * 3 + k) * 6 + (c_row * 3 + i)
                    
                    U[u_idx][col] = sign
                    V[v_idx][col] = sign
                    W[w_idx][col] = 1 # Wait, W should be 1?
                    # In tensor decomposition M = sum u v w,
                    # U, V, W are the factors.
                    # If I use schoolbook, U=1, V=1, W=1.
                    # But I am combining with signs.
                    # The sign is distributed to U and V?
                    # P = A B.
                    # If P = (sign * A) B, then U has sign.
                    # Or P = A (sign * B), then V has sign.
                    # Or P = sign * (A B), then W has sign?
                    # W is for C.
                    # C = sum P_t.
                    # So W should accumulate the coefficients.
                    # But in the tensor decomposition, W is fixed for the term.
                    # Here I am adding terms.
                    # So for each term, U, V, W are the factors.
                    # If the term is sign * A B, then U=sign, V=1, W=1?
                    # Or U=1, V=1, W=sign?
                    # It doesn't matter as long as product is sign.
                    # I'll put sign in U.
                    
                    U[u_idx][col] = sign
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1

    # Strassen terms
    # P1 = (A11 + A22)(B11 + B22)
    # A11 is block (0,0), A22 is block (1,1)
    # B11 is block (0,0), B22 is block (1,1)
    # C11 is block (0,0)
    
    # We need to add (A11 + A22) * (B11 + B22)
    # This expands to A11B11 + A11B22 + A22B11 + A22B22
    # But Strassen combines them into one multiplication.
    # The factors for the "sum" are not simply additive in U and V if we want to keep rank low.
    # Wait, Strassen's algorithm reduces the number of multiplications.
    # Each multiplication is a 3x3 matrix product.
    # So I should treat each P_t as a single 3x3 multiplication.
    # The inputs to the 3x3 multiplication are linear combinations of the blocks.
    # E.g., P1 input A is A11 + A22.
    # This means the U factors for P1 should correspond to A11 and A22.
    # But U is a matrix.
    # For the term corresponding to P1, the "A" part is A11 + A22.
    # So in the decomposition, the U factors for this term should have 1s for A11 indices and 1s for A22 indices.
    # Similarly for V.
    
    # Let's refine add_3x3_block to take a list of (block, coeff) for A and B.
    
    def add_strassen_term(a_blocks, b_blocks, c_blocks, offset):
        # a_blocks: list of (row, col, coeff)
        # b_blocks: list of (row, col, coeff)
        # c_blocks: list of (row, col, coeff)
        
        # We need to form the product of the linear combinations.
        # (sum ca * A_a) * (sum cb * B_b) = sum ca cb A_a B_b
        # This would expand to many terms.
        # But we want to treat it as ONE 3x3 multiplication.
        # The rank of a 3x3 multiplication is 27.
        # So we allocate 27 columns for this term.
        # The U factors for these 27 columns should encode the linear combination of A blocks.
        # The V factors should encode the linear combination of B blocks.
        # The W factors should encode the linear combination of C blocks.
        
        # For each of the 27 schoolbook terms of the 3x3 product:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = offset + (i * 3 + j) * 3 + k
                    
                    # U factors:
                    # A_part = sum (row, col, coeff) -> coeff * A[row, col]
                    # For the schoolbook term A[i, j] B[j, k],
                    # the A index is i, j.
                    # So we need to map i, j to the global A indices.
                    # The "virtual" A matrix is the sum of blocks.
                    # So U should have 1s (times coeff) at the positions corresponding to i, j in each block.
                    
                    for (ar, ac, ca) in a_blocks:
                        u_idx = (ar * 3 + i) * 3 + (ac * 3 + j)
                        U[u_idx][col] += ca
                        
                    for (br, bc, cb) in b_blocks:
                        v_idx = (br * 3 + j) * 3 + (bc * 3 + k)
                        V[v_idx][col] += cb
                        
                    for (cr, cc, cc_coeff) in c_blocks:
                        # C index mapping
                        # C_std[i, k] -> C_prompt[k, i]
                        # C block (cr, cc) has C_std indices i, k.
                        # Global C row = cr*3 + i, col = cc*3 + k
                        # Prompt index = col*6 + row = (cc*3 + k)*6 + (cr*3 + i)
                        w_idx = (cc * 3 + k) * 6 + (cr * 3 + i)
                        W[w_idx][col] += cc_coeff

    # Define Strassen terms
    # P1 = (A11 + A22)(B11 + B22) -> C11, C22
    # A blocks: (0,0,1), (1,1,1)
    # B blocks: (0,0,1), (1,1,1)
    # C blocks: (0,0,1), (1,1,1)
    add_strassen_term([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 0)
    
    # P2 = (A21 + A22)B11 -> C21, C22
    # A blocks: (1,0,1), (1,1,1)
    # B blocks: (0,0,1)
    # C blocks: (1,0,1), (1,1,-1)
    add_strassen_term([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)], 27)
    
    # P3 = A11(B12 - B22) -> C12, C22
    # A blocks: (0,0,1)
    # B blocks: (0,1,1), (1,1,-1)
    # C blocks: (0,1,1), (1,1,1)
    add_strassen_term([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)], 54)
    
    # P4 = A22(B21 - B11) -> C11, C21
    # A blocks: (1,1,1)
    # B blocks: (1,0,1), (0,0,-1)
    # C blocks: (0,0,1), (1,0,1)
    add_strassen_term([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], 81)
    
    # P5 = (A11 + A12)B22 -> C11, C12
    # A blocks: (0,0,1), (0,1,1)
    # B blocks: (1,1,1)
    # C blocks: (0,0,-1), (0,1,1)
    add_strassen_term([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)], 108)
    
    # P6 = (A21 - A11)(B11 + B12) -> C22
    # A blocks: (1,0,1), (0,0,-1)
    # B blocks: (0,0,1), (0,1,1)
    # C blocks: (1,1,1)
    add_strassen_term([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)], 135)
    
    # P7 = (A12 - A22)(B21 + B22) -> C11
    # A blocks: (0,1,1), (1,1,-1)
    # B blocks: (1,0,1), (1,1,1)
    # C blocks: (0,0,1)
    add_strassen_term([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)], 162)
    
    return {"U": U, "V": V, "W": W}
