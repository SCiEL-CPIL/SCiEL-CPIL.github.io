import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    
    # We use 2 Strassen 4x4x4 decompositions.
    # Rank per 4x4x4 is 49.
    # Total rank is 98.
    
    rank = 98
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to get Strassen 2x2x2 decomposition
    # Returns list of (u, v, w) vectors of length 4
    def strassen_2x2():
        # Strassen products for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We return the linear forms for A, B and the output vector W
        # A is 4x4? No, for the inner 2x2, A is 2x2 (4 elements).
        # Indices in 2x2: 0:00, 1:01, 2:10, 3:11
        
        # For each product, we have u (len 4), v (len 4), w (len 4)
        # But wait, the "scalar" product is just a number.
        # So u is a vector of coeffs for A elements.
        # v is a vector of coeffs for B elements.
        # w is a vector of coeffs for C elements.
        
        # M1
        u1 = [1, 0, 0, 1]
        v1 = [1, 1, 1, 1]
        # M1 contributes to C11, C12, C21, C22?
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So w1 has coeffs in positions 0 (C11), 3 (C22)
        # 0:1, 3:1
        w1 = [1, 0, 0, 1]
        
        # M2
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        # C21: +1, C22: -1
        w2 = [0, 0, 1, -1]
        
        # M3
        u3 = [1, 0, 0, 0]
        v3 = [0, 1, 0, -1]
        # C12: +1, C22: +1
        w3 = [0, 1, 0, 1]
        
        # M4
        u4 = [0, 0, 0, 1]
        v4 = [0, 0, 1, -1]
        # C11: +1, C21: +1
        w4 = [1, 0, 1, 0]
        
        # M5
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        # C11: -1, C12: +1
        w5 = [-1, 1, 0, 0]
        
        # M6
        u6 = [-1, 0, 1, 0]
        v6 = [1, 1, 0, 0]
        # C22: +1
        w6 = [0, 0, 0, 1]
        
        # M7
        u7 = [0, 1, 0, -1]
        v7 = [0, 0, 1, 1]
        # C11: +1
        w7 = [1, 0, 0, 0]
        
        return [(u1, v1, w1), (u2, v2, w2), (u3, v3, w3), (u4, v4, w4), (u5, v5, w5), (u6, v6, w6), (u7, v7, w7)]

    inner_decomp = strassen_2x2()
    
    # Top level Strassen for 4x4x4
    # We need to map the 2x2 blocks of the 4x4 matrix to the 4 elements of the "inner" matrix.
    # A (4x4) blocks: A11, A12, A21, A22.
    # A11 elements: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,4,5 in flattened 4x4?
    # Flatten A: A[i, j] -> index i*4 + j.
    # A11: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # A12: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    # A21: (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13
    # A22: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15
    
    # B (4x4) blocks: B11, B12, B21, B22.
    # Same indexing for B.
    
    # C (4x4) blocks: C11, C12, C21, C22.
    
    # For each top-level product Mk, we have linear forms in A-blocks and B-blocks.
    # We need to expand these to linear forms in A-elements and B-elements.
    
    # Let's define the top-level linear forms for A and B (length 4, for the 4 blocks)
    # And then map to length 16.
    
    top_u = [
        [1, 0, 0, 1], # M1: A11 + A22
        [0, 0, 1, 1], # M2: A21 + A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11 + A12
        [-1, 0, 1, 0], # M6: A21 - A11
        [0, 1, 0, -1]  # M7: A12 - A22
    ]
    
    top_v = [
        [1, 1, 1, 1], # M1: B11 + B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1], # M3: B12 - B22
        [0, 0, 1, -1], # M4: B21 - B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11 + B12
        [0, 0, 1, 1]  # M7: B21 + B22
    ]
    
    # Top level W coefficients for the 4 C-blocks
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    top_w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Mapping from block index (0..3) to element indices in 4x4
    block_map = {
        0: [0, 1, 4, 5],   # TL
        1: [2, 3, 6, 7],   # TR
        2: [8, 9, 12, 13], # BL
        3: [10, 11, 14, 15] # BR
    }
    
    col = 0
    for k in range(7):
        # Expand top_u[k] (len 4) to len 16
        u_full = [0.0] * 16
        for b_idx, coeff in enumerate(top_u[k]):
            if coeff != 0:
                for e_idx in block_map[b_idx]:
                    u_full[e_idx] = coeff
        
        # Expand top_v[k] (len 4) to len 32? No, B is 4x8.
        # Wait, for 4x4x4, B is 4x4.
        # We are doing 4x4x8.
        # So we split B into B_L (4x4) and B_R (4x4).
        # For B_L, indices 0..15. For B_R, indices 16..31.
        
        # We process B_L first (cols 0-7), then B_R (cols 8-15).
        # Rank is 2 * 49 = 98.
        
        for part in range(2):
            # part 0: B_L, part 1: B_R
            # For B_L, the 4x4 block map is the same.
            # For B_R, the indices are offset by 16.
            
            v_full = [0.0] * 32
            for b_idx, coeff in enumerate(top_v[k]):
                if coeff != 0:
                    for e_idx in block_map[b_idx]:
                        if part == 0:
                            v_full[e_idx] = coeff
                        else:
                            v_full[e_idx + 16] = coeff
            
            # Now we need to combine this with the inner Strassen decomposition.
            # The top-level product is M_k = (linear_A) (linear_B).
            # This is a 2x2 matrix product.
            # We decompose this 2x2 product using inner_strassen.
            # The inner_strassen gives 7 terms.
            # For each inner term l, we have u_inner (len 4), v_inner (len 4), w_inner (len 4).
            # u_inner corresponds to the 2x2 matrix X_k.
            # v_inner corresponds to the 2x2 matrix Y_k.
            # w_inner corresponds to the 2x2 matrix M_k.
            
            # The full U vector is u_full (from top) * u_inner?
            # No. The top-level linear form is a sum of A-blocks.
            # The inner linear form is a sum of elements within those blocks.
            # So the coefficient of A[i,j] is top_u[b_idx] * u_inner[e_idx_in_block].
            
            for l in range(7):
                u_inner, v_inner, w_inner = inner_decomp[l]
                
                U_row = [0.0] * 16
                V_row = [0.0] * 32
                W_row = [0.0] * 32 # C is 4x8, so 32 elements.
                
                # Fill U_row
                for b_idx in range(4):
                    if top_u[k][b_idx] != 0:
                        for i, e_idx in enumerate(block_map[b_idx]):
                            U_row[e_idx] = top_u[k][b_idx] * u_inner[i]
                
                # Fill V_row
                for b_idx in range(4):
                    if top_v[k][b_idx] != 0:
                        for i, e_idx in enumerate(block_map[b_idx]):
                            if part == 0:
                                V_row[e_idx] = top_v[k][b_idx] * v_inner[i]
                            else:
                                V_row[e_idx + 16] = top_v[k][b_idx] * v_inner[i]
                
                # Fill W_row
                # M_k is a 2x2 matrix.
                # w_inner has 4 elements corresponding to M_k[0,0], M_k[0,1], M_k[1,0], M_k[1,1].
                # These correspond to the 4 elements of the 2x2 block in C.
                # Which block of C?
                # The top-level product M_k contributes to specific C-blocks.
                # top_w[k] gives the coefficients for C11, C12, C21, C22.
                # For each C-block b_c_idx, if top_w[k][b_c_idx] is non-zero,
                # we add the contribution of w_inner to the corresponding elements in C.
                
                for b_c_idx in range(4):
                    coeff_c = top_w[k][b_c_idx]
                    if coeff_c != 0:
                        for i, e_idx in enumerate(block_map[b_c_idx]):
                            # C is 4x8.
                            # B_L corresponds to C_L (cols 0-3).
                            # B_R corresponds to C_R (cols 4-7).
                            # So if part==0, C elements are e_idx.
                            # If part==1, C elements are e_idx + 16.
                            if part == 0:
                                W_row[e_idx] = coeff_c * w_inner[i]
                            else:
                                W_row[e_idx + 16] = coeff_c * w_inner[i]
                
                # Assign to U, V, W
                for i in range(16):
                    U[i][col] = U_row[i]
                for i in range(32):
                    V[i][col] = V_row[i]
                    W[i][col] = W_row[i]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
