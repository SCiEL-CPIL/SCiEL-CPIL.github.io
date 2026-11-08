import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 97: Strassen 4x4x4 (49) + Schoolbook 4x4x3 (48)
    rank = 97
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4 * 4x4 (Left 4 columns of B) ---
    # Uses Kronecker product of two 2x2 Strassen decompositions.
    # Strassen 2x2x2 coefficients:
    # U (A-side):
    # M1: A11 + A22
    # M2: A21 + A22
    # M3: A11
    # M4: A22
    # M5: A11 + A12
    # M6: A21 - A11
    # M7: A22 - A12
    u_in = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    # V (B-side):
    # M1: B11 + B22
    # M2: B11
    # M3: B21 - B11
    # M4: B12 - B22
    # M5: B22
    # M6: B11 + B12
    # M7: B21 + B22
    v_in = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    # W (C-side reconstruction):
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    w_in = [
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ]
    
    col = 0
    # The 4x4x4 multiplication can be decomposed into 7x7 = 49 scalar multiplications
    # using the Kronecker product structure of Strassen.
    # We iterate over the 49 combinations of outer and inner Strassen steps.
    for k_outer in range(7):
        for l_inner in range(7):
            # U vector: 16 elements (4x4 matrix A)
            u_vec = [0] * 16
            # V vector: 28 elements (4x7 matrix B, only first 4 cols used here)
            v_vec = [0] * 28
            # W vector: 28 elements (4x7 matrix C, only first 4 cols used here)
            w_vec = [0] * 28
            
            # Calculate U and V contributions from outer and inner Strassen steps
            # The outer step operates on 2x2 blocks of the 4x4 matrix.
            # The inner step operates on the 2x2 elements within those blocks.
            
            # U: A is 4x4. Blocks are 2x2.
            # u_outer[k_outer] gives weights for the 4 blocks of A.
            # u_in[l_inner] gives weights for the 4 elements of a 2x2 block.
            
            u_out = u_in[k_outer] # Reusing u_in as the outer coefficient pattern is valid due to symmetry/structure in this specific Kronecker construction? 
            # Wait, the standard Kronecker product of Strassen uses the same coefficients for the outer and inner levels.
            # Let's verify the coefficients for the outer level.
            # Yes, for $M_{4,4,4} = M_{2,2,2} \otimes M_{2,2,2}$, the coefficients are the tensor product.
            # So u_outer = u_in, v_outer = v_in, w_outer = w_in.
            
            # Let's explicitly calculate the indices.
            # A is indexed by (i, j) with i,j in 0..3.
            # i = 2*i_b + i_i, j = 2*j_b + j_i.
            # Block index b corresponds to (i_b, j_b).
            # b=0: (0,0), b=1: (0,1), b=2: (1,0), b=3: (1,1).
            
            for i_b in range(2):
                for j_b in range(2):
                    b_idx = i_b * 2 + j_b
                    coeff_outer_u = u_in[k_outer][b_idx]
                    if coeff_outer_u == 0:
                        continue
                    for i_i in range(2):
                        for j_i in range(2):
                            coeff_inner_u = u_in[l_inner][i_i * 2 + j_i]
                            if coeff_inner_u == 0:
                                continue
                            total_coeff = coeff_outer_u * coeff_inner_u
                            i_global = 2 * i_b + i_i
                            j_global = 2 * j_b + j_i
                            u_vec[i_global * 4 + j_global] += total_coeff
            
            # V: B is 4x7. We only use the first 4 columns (indices 0-3).
            # B is indexed by (j, k) with j in 0..3, k in 0..6.
            # For the 4x4x4 part, k is in 0..3.
            # j = 2*j_b + j_i, k = 2*k_b + k_i.
            for j_b in range(2):
                for k_b in range(2):
                    b_idx = j_b * 2 + k_b
                    coeff_outer_v = v_in[k_outer][b_idx]
                    if coeff_outer_v == 0:
                        continue
                    for j_i in range(2):
                        for k_i in range(2):
                            coeff_inner_v = v_in[l_inner][j_i * 2 + k_i]
                            if coeff_inner_v == 0:
                                continue
                            total_coeff = coeff_outer_v * coeff_inner_v
                            j_global = 2 * j_b + j_i
                            k_global = 2 * k_b + k_i
                            v_vec[j_global * 7 + k_global] += total_coeff
            
            # W: C is 4x7. We only use the first 4 columns (indices 0-3).
            # C is indexed by (k, i) with k in 0..6, i in 0..3.
            # For the 4x4x4 part, k is in 0..3.
            # C = sum M_l * (u_l . A) (v_l . B)
            # The reconstruction W depends on the product of outer and inner reconstruction coefficients.
            # w_total[c_block_outer][c_elem_inner] = sum_{k_outer, l_inner} w_outer[k_outer][c_block_outer] * w_in[l_inner][c_elem_inner] * (u_outer[k_outer]... )
            # Actually, the W vector for a specific (k_outer, l_inner) pair is:
            # w_vec[index] = sum_{c_b, c_i} w_outer[k_outer][c_b] * w_in[l_inner][c_i]
            # where c_b is the block index in C (2x2 blocks of 2x2 elements)
            # and c_i is the element index within the block.
            
            for c_b in range(4):
                coeff_outer_w = w_in[k_outer][c_b]
                if coeff_outer_w == 0:
                    continue
                for c_i in range(4):
                    coeff_inner_w = w_in[l_inner][c_i]
                    if coeff_inner_w == 0:
                        continue
                    total_coeff = coeff_outer_w * coeff_inner_w
                    if total_coeff == 0:
                        continue
                    
                    # Map c_b (block) and c_i (element) to global indices
                    # c_b: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    c_b_row = c_b // 2
                    c_b_col = c_b % 2
                    
                    # c_i: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    c_i_row = c_i // 2
                    c_i_col = c_i % 2
                    
                    k_global = 2 * c_b_col + c_i_col
                    i_global = 2 * c_b_row + c_i_row
                    
                    w_vec[k_global * 4 + i_global] += total_coeff
            
            # Fill U, V, W columns
            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
            for r in range(28):
                W[r][col] = w_vec[r]
            
            col += 1
            
    # --- Part 2: 4x4 * 4x3 (Right 3 columns of B) ---
    # Schoolbook decomposition for the remaining 3 columns of B.
    # A is 4x4, B_right is 4x3.
    # C_right is 4x3.
    # Standard schoolbook: for each i, j, k_prime
    # C[i, k_prime] += A[i, j] * B[j, k_prime]
    # This contributes 1 to U[i*4+j], 1 to V[j*7+k_global], 1 to W[k_global*4+i]
    
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                k_global = 4 + k_prime
                U[i * 4 + j][col] = 1
                V[j * 7 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
