"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: 49 (for 4x4x4 Strassen) + 20 (for 4x1x5 outer product) = 69
    rank = 69
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 4x4x4 Strassen Decomposition ---
    # This handles A[0:4, 0:4] * B[0:4, 0:4] contributing to C[0:4, 0:4]
    
    # We represent linear forms as lists of coefficients for the 16 entries of A (4x4)
    # A indices: 0..15 (row-major: i*4+j)
    # B indices: 0..15 (row-major: j*4+k)
    # C indices: 0..15 (row-major: k*4+i) -- Note: C is 4x4 in this sub-problem
    
    # Strassen 2x2 factors (relative to local 2x2 blocks)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Local indices for 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_W = [
        [1, 0, 0, -1], # M1 contributes to C00, C11? 
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # W is a vector of length 4 for C00, C01, C10, C11
        # M1: C00=1, C11=1
        [1, 0, 0, 1],
        # M2: C10=1, C11=-1
        [0, 0, 1, -1],
        # M3: C01=1, C11=1
        [0, 1, 0, 1],
        # M4: C00=1, C10=1
        [1, 0, 1, 0],
        # M5: C00=-1, C01=1
        [-1, 1, 0, 0],
        # M6: C11=1
        [0, 0, 0, 1],
        # M7: C00=1
        [1, 0, 0, 0]
    ]
    
    # Top-level Strassen for 4x4
    # Blocks:
    # A00: A[0:2, 0:2] -> indices 0,1,4,5
    # A01: A[0:2, 2:4] -> indices 2,3,6,7
    # A10: A[2:4, 0:2] -> indices 8,9,12,13
    # A11: A[2:4, 2:4] -> indices 10,11,14,15
    
    # B00: B[0:2, 0:2] -> indices 0,1,4,5
    # B01: B[0:2, 2:4] -> indices 2,3,6,7
    # B10: B[2:4, 0:2] -> indices 8,9,12,13
    # B11: B[2:4, 2:4] -> indices 10,11,14,15
    
    # We need to map local 2x2 indices to global 4x4 indices
    def map_A(local_idx, block):
        # local_idx: 0..3 (00, 01, 10, 11)
        # block: 0..3 (00, 01, 10, 11)
        r = local_idx // 2
        c = local_idx % 2
        br = block // 2
        bc = block % 2
        global_r = br * 2 + r
        global_c = bc * 2 + c
        return global_r * 4 + global_c
        
    def map_B(local_idx, block):
        r = local_idx // 2
        c = local_idx % 2
        br = block // 2
        bc = block % 2
        global_r = br * 2 + r
        global_c = bc * 2 + c
        return global_r * 4 + global_c
        
    def map_C(local_idx, block):
        # C is 4x4, indices k*4+i
        # Local C is 2x2, indices l_k*2 + l_i
        # Block C is 2x2, indices b_k*2 + b_i
        # Global C index: (b_k*2 + l_k) * 4 + (b_i*2 + l_i)
        # local_idx: 0..3 (00, 01, 10, 11) where 00 is C00, 01 is C01, 10 is C10, 11 is C11
        # In local 2x2, row is k, col is i? 
        # Standard C is row i, col k? No, C[i, k].
        # In the tensor decomposition, W index is k*n + i.
        # So for 4x4, W index is k*4 + i.
        # Local 2x2 C: W index is l_k*2 + l_i.
        # Block C: b_k, b_i.
        # Global k = b_k*2 + l_k, Global i = b_i*2 + l_i.
        # Global W index = (b_k*2 + l_k)*4 + (b_i*2 + l_i).
        
        l_k = local_idx // 2
        l_i = local_idx % 2
        b_k = block // 2
        b_i = block % 2
        global_k = b_k * 2 + l_k
        global_i = b_i * 2 + l_i
        return global_k * 4 + global_i

    # Top-level Strassen combinations
    # M1_top = (A00+A11)(B00+B11)
    # M2_top = (A10+A11)B00
    # M3_top = A00(B01-B11)
    # M4_top = A11(B10-B00)
    # M5_top = (A00+A01)B11
    # M6_top = (A10-A00)(B00+B01)
    # M7_top = (A01-A11)(B10+B11)
    
    # For each top-level product, we have X and Y (2x2 matrices of linear forms)
    # We iterate through the 7 top-level products
    
    # Define the linear forms for X and Y for each top-level product
    # X_k is 2x2, Y_k is 2x2
    # We represent X_k as a list of 4 linear forms (for X00, X01, X10, X11)
    # Each linear form is a dict {global_A_idx: coeff}
    
    # Helper to add to a linear form
    def add_to_lf(lf, idx, coeff):
        if coeff == 0: return
        lf[idx] = lf.get(idx, 0) + coeff
        if lf[idx] == 0:
            del lf[idx]

    top_products = []
    
    # M1: X = A00+A11, Y = B00+B11
    X1 = [{} for _ in range(4)]
    Y1 = [{} for _ in range(4)]
    for l in range(4):
        add_to_lf(X1[l], map_A(l, 0), 1)
        add_to_lf(X1[l], map_A(l, 3), 1)
        add_to_lf(Y1[l], map_B(l, 0), 1)
        add_to_lf(Y1[l], map_B(l, 3), 1)
    top_products.append((X1, Y1))
    
    # M2: X = A10+A11, Y = B00
    X2 = [{} for _ in range(4)]
    Y2 = [{} for _ in range(4)]
    for l in range(4):
        add_to_lf(X2[l], map_A(l, 2), 1)
        add_to_lf(X2[l], map_A(l, 3), 1)
        add_to_lf(Y2[l], map_B(l, 0), 1)
    top_products.append((X2, Y2))
    
    # M3: X = A00, Y = B01-B11
    X3 = [{} for _ in range(4)]
    Y3 = [{} for _ in range(4)]
    for l in range(4):
        add_to_lf(X3[l], map_A(l, 0), 1)
        add_to_lf(Y3[l], map_B(l, 1), 1)
        add_to_lf(Y3[l], map_B(l, 3), -1)
    top_products.append((X3, Y3))
    
    # M4: X = A11, Y = B10-B00
    X4 = [{} for _ in range(4)]
    Y4 = [{} for _ in range(4)]
    for l in range(4):
        add_to_lf(X4[l], map_A(l, 3), 1)
        add_to_lf(Y4[l], map_B(l, 2), 1)
        add_to_lf(Y4[l], map_B(l, 0), -1)
    top_products.append((X4, Y4))
    
    # M5: X = A00+A01, Y = B11
    X5 = [{} for _ in range(4)]
    Y5 = [{} for _ in range(4)]
    for l in range(4):
        add_to_lf(X5[l], map_A(l, 0), 1)
        add_to_lf(X5[l], map_A(l, 1), 1)
        add_to_lf(Y5[l], map_B(l, 3), 1)
    top_products.append((X5, Y5))
    
    # M6: X = A10-A00, Y = B00+B01
    X6 = [{} for _ in range(4)]
    Y6 = [{} for _ in range(4)]
    for l in range(4):
        add_to_lf(X6[l], map_A(l, 2), 1)
        add_to_lf(X6[l], map_A(l, 0), -1)
        add_to_lf(Y6[l], map_B(l, 0), 1)
        add_to_lf(Y6[l], map_B(l, 1), 1)
    top_products.append((X6, Y6))
    
    # M7: X = A01-A11, Y = B10+B11
    X7 = [{} for _ in range(4)]
    Y7 = [{} for _ in range(4)]
    for l in range(4):
        add_to_lf(X7[l], map_A(l, 1), 1)
        add_to_lf(X7[l], map_A(l, 3), -1)
        add_to_lf(Y7[l], map_B(l, 2), 1)
        add_to_lf(Y7[l], map_B(l, 3), 1)
    top_products.append((X7, Y7))
    
    # Now, for each top product k (0..6), we compute the 7 inner Strassen factors
    for k_idx in range(7):
        X_k, Y_k = top_products[k_idx]
        
        # For each inner factor i (0..6)
        for i_idx in range(7):
            # Get U_i, V_i, W_i for the inner 2x2 Strassen
            u_i = strassen_U[i_idx]
            v_i = strassen_V[i_idx]
            w_i = strassen_W[i_idx]
            
            # Compute U_new: linear form in A
            U_new = {}
            for l in range(4):
                coeff_u = u_i[l]
                if coeff_u == 0: continue
                # X_k[l] is a linear form in A
                for a_idx, coeff_a in X_k[l].items():
                    add_to_lf(U_new, a_idx, coeff_u * coeff_a)
                    
            # Compute V_new: linear form in B
            V_new = {}
            for l in range(4):
                coeff_v = v_i[l]
                if coeff_v == 0: continue
                # Y_k[l] is a linear form in B
                for b_idx, coeff_b in Y_k[l].items():
                    add_to_lf(V_new, b_idx, coeff_v * coeff_b)
            
            # Compute W_new: contributions to C
            # w_i is a vector of 4 coeffs for C00, C01, C10, C11 of Z_k
            # Z_k is the k-th top-level product.
            # We need to map these to the final C.
            
            # Top-level Strassen combination for C:
            # C00 = Z1 + Z4 - Z5 + Z7
            # C01 = Z3 + Z5
            # C10 = Z2 + Z4
            # C11 = Z1 - Z2 + Z3 + Z6
            
            # Determine which blocks of C Z_k contributes to
            # Z_k corresponds to top product index k_idx (0-based)
            # k_idx 0 -> Z1, 1 -> Z2, ..., 6 -> Z7
            
            c_blocks = [0, 0, 0, 0] # Coeffs for C00, C01, C10, C11
            
            if k_idx == 0: # Z1
                c_blocks[0] = 1
                c_blocks[3] = 1
            elif k_idx == 1: # Z2
                c_blocks[2] = 1
                c_blocks[3] = -1
            elif k_idx == 2: # Z3
                c_blocks[1] = 1
                c_blocks[3] = 1
            elif k_idx == 3: # Z4
                c_blocks[0] = 1
                c_blocks[2] = 1
            elif k_idx == 4: # Z5
                c_blocks[0] = -1
                c_blocks[1] = 1
            elif k_idx == 5: # Z6
                c_blocks[3] = 1
            elif k_idx == 6: # Z7
                c_blocks[0] = 1
            
            # For each local C entry (l_c in 0..3), add contribution
            for l_c in range(4):
                coeff_w = w_i[l_c]
                coeff_block = c_blocks[l_c]
                if coeff_w == 0 or coeff_block == 0: continue
                
                final_coeff = coeff_w * coeff_block
                if final_coeff == 0: continue
                
                # Map local C index l_c in block l_c_block to global C index
                # Wait, l_c is the local index within the 2x2 Z_k.
                # The block of C that Z_k corresponds to is determined by l_c?
                # No. Z_k is a 2x2 matrix.
                # The entries of Z_k are added to specific 2x2 blocks of C.
                # Specifically, Z_k is added to C00, C01, etc. with coefficients c_blocks.
                # But Z_k itself is 2x2.
                # The entry Z_k[0,0] (l_c=0) is added to C00[0,0], C00[0,1], C00[1,0], C00[1,1]?
                # No. Z_k is the result of a 2x2 multiplication.
                # Z_k[0,0] is a scalar.
                # It contributes to C[0,0] (global) with coefficient c_blocks[0].
                # It contributes to C[0,1] (global) with coefficient c_blocks[0]?
                # No.
                # Let's re-read Strassen.
                # C = AB.
                # C00 (2x2 block) = Z1 + Z4 - Z5 + Z7.
                # Here Z1, Z4, Z5, Z7 are 2x2 matrices.
                # So Z1[0,0] is added to C00[0,0].
                # Z1[0,1] is added to C00[0,1].
                # Z1[1,0] is added to C00[1,0].
                # Z1[1,1] is added to C00[1,1].
                
                # So for a given l_c (0..3), which block of C does it go to?
                # l_c=0 (00) -> Block 00 (C00)
                # l_c=1 (01) -> Block 01 (C01)
                # l_c=2 (10) -> Block 10 (C10)
                # l_c=3 (11) -> Block 11 (C11)
                
                # So the target block is l_c.
                # And the coefficient is c_blocks[l_c].
                
                target_block = l_c
                target_coeff = c_blocks[target_block]
                
                if target_coeff == 0: continue
                
                # Map local C index l_c in block target_block to global C index
                global_c_idx = map_C(l_c, target_block)
                
                add_to_lf(W_new, global_c_idx, final_coeff)

            # Now we have U_new, V_new, W_new for this scalar multiplication
            # Add to U, V, W matrices
            
            for a_idx, coeff in U_new.items():
                U[a_idx][col] = coeff
            for b_idx, coeff in V_new.items():
                V[b_idx][col] = coeff
            for c_idx, coeff in W_new.items():
                W[c_idx][col] = coeff
                
            col += 1
            
    # --- Part 2: 4x1x5 Outer Product ---
    # A[0:4, 4] * B[4, 0:5] -> C[0:4, 0:5]
    # A indices: i*5 + 4
    # B indices: 4*5 + j = 20 + j
    # C indices: j*4 + i
    
    for i in range(4):
        for j in range(5):
            u_idx = i * 5 + 4
            v_idx = 20 + j
            w_idx = j * 4 + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
