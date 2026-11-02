import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct Strassen decomposition for 2x2x2 (Rank 7)
    # U2, V2, W2 are 4x7 matrices
    # Indices for 2x2: A is 2x2 (rows 0,1,2,3), B is 2x2 (rows 0,1,2,3), C is 2x2 (rows 0,1,2,3)
    # U2[i, c] corresponds to A[i//2, i%2]
    # V2[j, c] corresponds to B[j//2, j%2]
    # W2[k, c] corresponds to C[k//2, k%2]
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # P1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1; W2[1, 0] = 1; W2[2, 0] = 1 # C00, C11, C01, C10? 
    # Wait, W2 maps products to C entries.
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's fill W2 based on C formulas
    # W2[c, r] is coefficient of product r in C_c
    
    # C00 (idx 0): P1(1) + P4(1) - P5(1) + P7(1)
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    
    # C01 (idx 1): P3(1) + P5(1)
    W2[1, 2] = 1; W2[1, 4] = 1
    
    # C10 (idx 2): P2(1) + P4(1)
    W2[2, 1] = 1; W2[2, 3] = 1
    
    # C11 (idx 3): P1(1) - P2(1) + P3(1) + P6(1)
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1
    
    # Now fill U2 and V2 for each product P_r
    # P1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    
    # P2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    
    # P3: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    
    # P4: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    
    # P5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    
    # P6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    
    # P7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # 2. Construct Strassen decomposition for 4x4x4 (Rank 49)
    # U4, V4, W4 are 16x49 matrices
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49)) # Wait, V is m*p. For 4x4x4, m=4, p=4, so 16x49.
    W4 = np.zeros((16, 49)) # W is p*n. For 4x4x4, p=4, n=4, so 16x49.
    
    # The 7 products K_r are 2x2 matrix multiplications.
    # Each K_r uses U2, V2, W2.
    # We need to map the 2x2 block indices to the 4x4 matrix indices.
    
    # Block mapping for A (4x4):
    # A00: rows 0,1; cols 0,1 -> U indices 0,1,4,5
    # A01: rows 0,1; cols 2,3 -> U indices 2,3,6,7
    # A10: rows 2,3; cols 0,1 -> U indices 8,9,12,13
    # A11: rows 2,3; cols 2,3 -> U indices 10,11,14,15
    
    # Block mapping for B (4x4):
    # B00: rows 0,1; cols 0,1 -> V indices 0,1,4,5
    # B01: rows 0,1; cols 2,3 -> V indices 2,3,6,7
    # B10: rows 2,3; cols 0,1 -> V indices 8,9,12,13
    # B11: rows 2,3; cols 2,3 -> V indices 10,11,14,15
    
    # Block mapping for C (4x4):
    # C00: rows 0,1; cols 0,1 -> W indices 0,1,4,5
    # C01: rows 0,1; cols 2,3 -> W indices 2,3,6,7
    # C10: rows 2,3; cols 0,1 -> W indices 8,9,12,13
    # C11: rows 2,3; cols 2,3 -> W indices 10,11,14,15
    
    # Define the 7 Strassen products for M4
    # Each product is a 2x2 matrix multiplication A' B'
    # We define A' and B' as linear combinations of blocks.
    
    products = [
        # (A_coeffs, B_coeffs)
        # A_coeffs: list of (block_idx, coeff)
        # B_coeffs: list of (block_idx, coeff)
        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
        
        # K1 = (A00+A11)(B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # K2 = (A10+A11)B00
        ([(2, 1), (3, 1)], [(0, 1)]),
        # K3 = A00(B01-B11)
        ([(0, 1)], [(1, 1), (3, -1)]),
        # K4 = A11(B10-B00)
        ([(3, 1)], [(2, 1), (0, -1)]),
        # K5 = (A00+A01)B11
        ([(0, 1), (1, 1)], [(3, 1)]),
        # K6 = (A10-A00)(B00+B01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # K7 = (A01-A11)(B10+B11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])
    ]
    
    # C coefficients for K_r
    # C00 = K1 + K4 - K5 + K7
    # C01 = K3 + K5
    # C10 = K2 + K4
    # C11 = K1 - K2 + K3 + K6
    c_coeffs = [
        [(0, 1), (3, 1), (4, -1), (6, 1)], # C00
        [(2, 1), (4, 1)],                  # C01
        [(1, 1), (3, 1)],                  # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)]  # C11
    ]
    
    # Helper to get block indices
    def get_block_indices(block_idx, is_row_based=True):
        # Returns list of 4 indices in the 16-dim vector
        # Block 0: 0,1,4,5
        # Block 1: 2,3,6,7
        # Block 2: 8,9,12,13
        # Block 3: 10,11,14,15
        if is_row_based:
            # For U (A) and V (B), the index is i*m+j or j*p+k
            # A (U): i is row, j is col.
            # Block 0 (A00): i in 0,1; j in 0,1. Indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
            # Block 1 (A01): i in 0,1; j in 2,3. Indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7.
            # Block 2 (A10): i in 2,3; j in 0,1. Indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13.
            # Block 3 (A11): i in 2,3; j in 2,3. Indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15.
            return [block_idx * 4 + 0, block_idx * 4 + 1, block_idx * 4 + 4, block_idx * 4 + 5]
        else:
            # For W (C), the index is k*n+i. k is row, i is col.
            # Block 0 (C00): k in 0,1; i in 0,1. Indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
            # Block 1 (C01): k in 0,1; i in 2,3. Indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7.
            # Block 2 (C10): k in 2,3; i in 0,1. Indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13.
            # Block 3 (C11): k in 2,3; i in 2,3. Indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15.
            return [block_idx * 4 + 0, block_idx * 4 + 1, block_idx * 4 + 4, block_idx * 4 + 5]
            
    # Note: The mapping for U and V is slightly different in terms of how blocks are laid out if we consider them as 2x2 submatrices of a 4x4.
    # A (4x4):
    # A00 A01
    # A10 A11
    # A00 is top-left.
    # U indices: i*4+j.
    # A00: i=0,1; j=0,1 -> 0,1,4,5.
    # A01: i=0,1; j=2,3 -> 2,3,6,7.
    # A10: i=2,3; j=0,1 -> 8,9,12,13.
    # A11: i=2,3; j=2,3 -> 10,11,14,15.
    # This matches get_block_indices(0..3).
    
    # B (4x4):
    # B00 B01
    # B10 B11
    # V indices: j*4+k.
    # B00: j=0,1; k=0,1 -> 0,1,4,5.
    # B01: j=0,1; k=2,3 -> 2,3,6,7.
    # B10: j=2,3; k=0,1 -> 8,9,12,13.
    # B11: j=2,3; k=2,3 -> 10,11,14,15.
    # This also matches get_block_indices(0..3).
    
    # C (4x4):
    # C00 C01
    # C10 C11
    # W indices: k*4+i.
    # C00: k=0,1; i=0,1 -> 0,1,4,5.
    # C01: k=0,1; i=2,3 -> 2,3,6,7.
    # C10: k=2,3; i=0,1 -> 8,9,12,13.
    # C11: k=2,3; i=2,3 -> 10,11,14,15.
    # This also matches get_block_indices(0..3).
    
    # Now, for each of the 7 products r (0..6), we add the contribution of U2, V2, W2
    for r in range(7):
        a_coeffs, b_coeffs = products[r]
        base_col = r * 7
        
        # Add U2 contribution
        for local_row in range(4):
            # U2[local_row, c] is non-zero for c in 0..6
            # We need to map local_row (0..3 of 2x2) to the physical rows of A'
            # The local_row corresponds to the row in the 2x2 block.
            # But A' is a sum of blocks.
            # U2[local_row, c] multiplies A'[local_row//2, local_row%2]?
            # No, U2 is defined for a single 2x2 matrix.
            # The rows of U2 correspond to the entries of the 2x2 matrix A'.
            # Entry (lr//2, lr%2) of A' is a linear combination of entries of A.
            
            # Wait, U2[local_row, c] is the coefficient for A'[lr//2, lr%2] in the c-th product.
            # So we need to find which physical rows of A contribute to A'[lr//2, lr%2].
            
            # Let's iterate over the blocks in a_coeffs
            # For each block b with coeff cb, the entry A'[lr//2, lr%2] gets contribution from A_b[local_row_in_block]
            # The local row in the 2x2 block A_b is (lr//2, lr%2).
            # The physical index of A_b[ir, ic] is get_block_indices(b)[ir*2+ic]?
            # No, get_block_indices returns 4 indices for the 4 entries of the 2x2 block.
            # The order in get_block_indices is:
            # idx 0: (0,0)
            # idx 1: (0,1)
            # idx 2: (1,0)
            # idx 3: (1,1)
            # So local entry (lr//2, lr%2) corresponds to index (lr//2)*2 + (lr%2) in the list of 4.
            
            local_entry_idx = (lr // 2) * 2 + (lr % 2)
            
            for b_idx, cb in a_coeffs:
                phys_idx = get_block_indices(b_idx)[local_entry_idx]
                U4[phys_idx, base_col + local_row] += U2[local_row, 0] * cb # Wait, U2[local_row, c] varies with c.
                # We are filling column base_col + c.
                # The loop over c is implicit? No, U2 has 7 columns.
                # We need to add U2[:, c] to U4[phys_indices, base_col + c]
        
        # Let's re-structure the loop to be cleaner.
        # For each product r (0..6):
        #   For each local column c (0..6):
        #     For each local row lr (0..3):
        #       val = U2[lr, c]
        #       if val != 0:
        #         Find physical rows in A that correspond to A'[lr//2, lr%2]
        #         Add val * coeff to U4[phys_row, base_col + c]
        
        # Let's clear U4, V4, W4 and do it properly.
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        for r in range(7):
            a_coeffs, b_coeffs = products[r]
            base_col = r * 7
            
            # Build mapping for A'
            # A'[lr] is linear combo of A blocks
            # We need to map each of the 4 local rows of A' to physical rows of A
            a_map = []
            for lr in range(4):
                local_entry_idx = (lr // 2) * 2 + (lr % 2)
                entries = []
                for b_idx, cb in a_coeffs:
                    phys_idx = get_block_indices(b_idx)[local_entry_idx]
                    entries.append((phys_idx, cb))
                a_map.append(entries)
                
            # Build mapping for B'
            b_map = []
            for lr in range(4):
                local_entry_idx = (lr // 2) * 2 + (lr % 2)
                entries = []
                for b_idx, cb in b_coeffs:
                    phys_idx = get_block_indices(b_idx)[local_entry_idx]
                    entries.append((phys_idx, cb))
                b_map.append(entries)
            
            # Fill U4 and V4
            for c in range(7):
                for lr in range(4):
                    val_u = U2[lr, c]
                    if val_u != 0:
                        for phys_idx, cb in a_map[lr]:
                            U4[phys_idx, base_col + c] += val_u * cb
                            
                    val_v = V2[lr, c]
                    if val_v != 0:
                        for phys_idx, cb in b_map[lr]:
                            V4[phys_idx, base_col + c] += val_v * cb
                            
        # Fill W4
        # W4 maps products to C blocks.
        # C_c is linear combo of K_r.
        # K_r is a 2x2 matrix.
        # W2 maps local products of K_r to local entries of K_r.
        # We need to map local entries of K_r to physical entries of C.
        
        # For each C block c_block (0..3):
        #   For each local entry le (0..3) in the 2x2 block:
        #     For each product r contributing to c_block:
        #       For each local product c (0..6) in W2:
        #         val = W2[le, c]
        #         if val != 0:
        #           phys_idx = get_block_indices(c_block)[le]
        #           W4[phys_idx, r*7 + c] += val * c_coeff
           
        for c_block in range(4):
            coeffs = c_coeffs[c_block]
            for le in range(4):
                phys_idx = get_block_indices(c_block)[le]
                for r, cr in coeffs:
                    base_col = r * 7
                    for c in range(7):
                        val = W2[le, c]
                        if val != 0:
                            W4[phys_idx, base_col + c] += val * cr

    # 3. Construct 4x4x8 decomposition (Rank 98)
    # U_global (16, 98), V_global (32, 98), W_global (32, 98)
    
    U_global = np.zeros((16, 98))
    V_global = np.zeros((32, 98))
    W_global = np.zeros((32, 98))
    
    # Copy 1: B1 (cols 0-3), C1 (cols 0-3)
    # U1 = U4
    U_global[:, 0:49] = U4
    
    # V1: Map V4 (16x49) to V_global (32x49)
    # V4 row j*4+k (k in 0..3) maps to V_global row j*8+k
    for j in range(4):
        for k in range(4):
            v4_idx = j * 4 + k
            v_global_idx = j * 8 + k
            V_global[v_global_idx, 0:49] = V4[v4_idx, 0:49]
            
    # W1: Map W4 (16x49) to W_global (32x49)
    # W4 row k*4+i (k in 0..3) maps to W_global row k*4+i
    # Since k is the same (0..3), the indices are identical.
    W_global[0:16, 0:49] = W4
    
    # Copy 2: B2 (cols 4-7), C2 (cols 4-7)
    # U2 = U4
    U_global[:, 49:98] = U4
    
    # V2: Map V4 (16x49) to V_global (32x49) offset by 49
    # V4 row j*4+k' (k' in 0..3) maps to V_global row j*8 + (k'+4)
    for j in range(4):
        for k in range(4):
            v4_idx = j * 4 + k
            v_global_idx = j * 8 + (k + 4)
            V_global[v_global_idx, 49:98] = V4[v4_idx, 0:49]
            
    # W2: Map W4 (16x49) to W_global (32x49) offset by 49
    # W4 row k'*4+i (k' in 0..3) maps to W_global row (k'+4)*4 + i
    for k in range(4):
        for i in range(4):
            w4_idx = k * 4 + i
            w_global_idx = (k + 4) * 4 + i
            W_global[w_global_idx, 49:98] = W4[w4_idx, 0:49]

    return {"U": U_global, "V": V_global, "W": W_global}
# EVOLVE-BLOCK-END
