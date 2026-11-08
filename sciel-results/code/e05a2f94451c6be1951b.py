import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    
    # Target rank is 98 (2 * 49)
    # 49 is the rank of the 4x4x4 Strassen decomposition (2^7? No, 7^2=49).
    # Wait, Strassen for 2x2 is rank 7.
    # For 4x4, we can use 4 blocks of 2x2? No.
    # We can use a block Strassen decomposition.
    # A 4x4 matrix multiplication can be decomposed into 7 multiplications of 2x2 blocks?
    # Yes, if we group the 4x4 matrix into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # The 7 Strassen products P0..P6 are 2x2 matrix multiplications.
    # Each 2x2 multiplication requires rank 7.
    # Total rank = 7 * 7 = 49.
    # For 4x4 * 4x8, we have two 4x4 * 4x4 multiplications (splitting B into B1 and B2).
    # Total rank = 49 + 49 = 98.
    
    rank = 98
    
    # --- Step 1: Define 2x2x2 Strassen factors ---
    # U_2: 4x7, V_2: 4x7, W_2: 4x7
    # Indices: 0:00, 1:01, 2:10, 3:11
    
    U_2 = np.array([
        [ 1,  0,  1,  0,  1, -1,  0], # 00
        [ 0,  0,  0,  0,  1,  0,  1], # 01
        [ 0,  1,  0,  0,  0,  1,  0], # 10
        [ 1,  1,  0,  1,  0,  0, -1]  # 11
    ])
    
    V_2 = np.array([
        [ 1,  1,  0, -1,  0,  1,  0], # 00
        [ 0,  0,  1,  0,  0,  1,  0], # 01
        [ 0,  0,  0,  1,  0,  0,  1], # 10
        [ 1,  0, -1,  0,  1,  0,  1]  # 11
    ])
    
    W_2 = np.array([
        [ 1,  0,  0,  1, -1,  0,  1], # 00
        [ 0,  0,  1,  0,  1,  0,  0], # 01
        [ 0,  1,  0,  1,  0,  0,  0], # 10
        [ 1, -1,  1,  0,  0,  1,  0]  # 11
    ])
    
    # --- Step 2: Construct 4x4x4 factors from 2x2x2 factors ---
    # The 4x4 Strassen decomposition is built by applying the 2x2 Strassen
    # to the block structure.
    # A_eff = linear_combination_of_blocks(A)
    # This is equivalent to a Kronecker-like expansion but with the Strassen logic applied at the block level.
    
    # Let's explicitly build the 49 rank terms.
    # The 49 terms come from 7 "super-strassen" steps, each of which is a 2x2 Strassen (7 terms).
    
    # Specs for the 7 super-steps of 4x4x4
    # A_specs[k]: list of (coeff, block_idx)
    # B_specs[k]: list of (coeff, block_idx)
    # C_specs[k]: list of (coeff, block_idx)
    # Block indices: 0:TL, 1:TR, 2:BL, 3:BR
    
    A_specs = [
        [(1, 0), (1, 3)], # P0: A00+A11
        [(1, 2), (1, 3)], # P1: A10+A11
        [(1, 0)],         # P2: A00
        [(1, 3)],         # P3: A11
        [(1, 0), (1, 1)], # P4: A00+A01
        [(1, 2), (-1, 0)],# P5: A10-A00
        [(1, 1), (-1, 3)] # P6: A01-A11
    ]
    
    B_specs = [
        [(1, 0), (1, 3)], # P0: B00+B11
        [(1, 0)],         # P1: B00
        [(1, 1), (-1, 3)],# P2: B01-B11
        [(1, 2), (-1, 0)],# P3: B10-B00
        [(1, 3)],         # P4: B11
        [(1, 0), (1, 1)], # P5: B00+B01
        [(1, 2), (1, 3)]  # P6: B10+B11
    ]
    
    C_specs = [
        [(1, 0), (1, 3)], # P0: C00+C11
        [(1, 2), (-1, 3)],# P1: C10-C11
        [(1, 1), (1, 3)], # P2: C01+C11
        [(1, 0), (1, 2)], # P3: C00+C10
        [(-1, 0), (1, 1)],# P4: -C00+C01
        [(1, 3)],         # P5: C11
        [(1, 0)]          # P6: C00
    ]
    
    # Initialize 4x4x4 factors
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    # Map local 2x2 index to global 4x4 index
    # Global index = (block_row * 2 + r) * 4 + (block_col * 2 + c)
    # Local index = r * 2 + c
    
    def get_global_idx(block_idx, r, c, dim=4):
        br, bc = divmod(block_idx, 2)
        # For U (A): row is i, col is j. Block is (br, bc).
        # For V (B): row is j, col is k. Block is (br, bc).
        # For W (C): row is k, col is i. Block is (br, bc).
        # In all cases for the "input" matrices A and B, the block index maps to (row_block, col_block).
        # For W, the block index maps to (row_block, col_block) of C.
        # C is p*n. W index is k*n + i.
        # C blocks are 2x2 submatrices of C.
        # C row index is k, col index is i.
        # So block (br, bc) corresponds to k in [br*2, br*2+1] and i in [bc*2, bc*2+1].
        # Global W index = k*4 + i.
        
        global_row = br * 2 + r
        global_col = bc * 2 + c
        return global_row * dim + global_col

    # Construct 4x4x4 factors
    for k_p in range(7):
        for m_s in range(7):
            col = k_p * 7 + m_s
            
            # U (A)
            for coeff, block_idx in A_specs[k_p]:
                br, bc = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        g_idx = get_global_idx(block_idx, r, c, 4)
                        l_idx = r * 2 + c
                        U_4[g_idx, col] += coeff * U_2[l_idx, m_s]
            
            # V (B)
            for coeff, block_idx in B_specs[k_p]:
                br, bc = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        g_idx = get_global_idx(block_idx, r, c, 4)
                        l_idx = r * 2 + c
                        V_4[g_idx, col] += coeff * V_2[l_idx, m_s]
            
            # W (C)
            for coeff, block_idx in C_specs[k_p]:
                br, bc = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        # C is 4x4. W index is k*4+i.
                        # Block (br, bc) -> k in br, i in bc.
                        g_idx = get_global_idx(block_idx, r, c, 4)
                        l_idx = r * 2 + c
                        W_4[g_idx, col] += coeff * W_2[l_idx, m_s]

    # --- Step 3: Construct 4x4x8 factors ---
    # B is 4x8. We split it into B1 (cols 0-3) and B2 (cols 4-7).
    # C is 4x8. We split it into C1 (cols 0-3) and C2 (cols 4-7).
    # The multiplication is [C1 C2] = A * [B1 B2] = [A*B1, A*B2].
    # We perform two independent 4x4x4 multiplications.
    # Rank = 49 (for B1) + 49 (for B2) = 98.
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # First half: B1 (cols 0-3 of B, cols 0-3 of C)
    U[:, 0:49] = U_4
    
    # V1 corresponds to B1 (j*8 + k, where k in 0-3)
    # V_4 indices are j*4 + k_local (k_local in 0-3).
    # We need to map j*4 + k_local to j*8 + k_local.
    # V_4 is 16x49. Row index r in V_4 is j*4 + k_local.
    # We create V1 as 32x49.
    V1 = np.zeros((32, 49))
    for j in range(4):
        for k in range(4):
            V1[j*8 + k, :] = V_4[j*4 + k, :]
    V[:, 0:49] = V1
    
    # W1 corresponds to C1 (k*4 + i, where k in 0-3)
    # W_4 indices are k*4 + i (k in 0-3).
    # We need to map k*4 + i to k*4 + i (same indexing for C1 part of W).
    # Wait, W is 32x98. W index is k*4 + i.
    # For C1, k is 0-3. So rows 0-15 of W correspond to C1.
    # W_4 is 16x49. Rows 0-15 correspond to C1.
    W[0:16, 0:49] = W_4
    
    # Second half: B2 (cols 4-7 of B, cols 4-7 of C)
    U[:, 49:98] = U_4
    
    # V2 corresponds to B2 (j*8 + k, where k in 4-7)
    # V_4 indices are j*4 + k_local (k_local in 0-3).
    # We need to map j*4 + k_local to j*8 + (k_local + 4).
    V2 = np.zeros((32, 49))
    for j in range(4):
        for k in range(4):
            V2[j*8 + (k + 4), :] = V_4[j*4 + k, :]
    V[:, 49:98] = V2
    
    # W2 corresponds to C2 (k*4 + i, where k in 4-7)
    # W_4 indices are k*4 + i (k in 0-3).
    # We need to map k*4 + i to (k+4)*4 + i = k*4 + 16 + i.
    # So rows 16-31 of W correspond to C2.
    W[16:32, 49:98] = W_4
    
    # Time check
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
