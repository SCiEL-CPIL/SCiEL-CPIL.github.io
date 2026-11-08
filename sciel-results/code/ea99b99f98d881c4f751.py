import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # 1. Generate Strassen 2x2x2 decomposition
    # Rank 7
    # U is 4x7, V is 4x7, W is 4x7
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # M1: (A11+A22)(B11+B22)
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1 # C11
    W_222[3, 0] = 1 # C22
    
    # M2: (A21+A22)B11
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1 # C21
    W_222[3, 1] = -1 # C22
    
    # M3: A11(B12-B22)
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[1, 2] = 1 # C12
    W_222[3, 2] = 1 # C22
    
    # M4: A22(B21-B11)
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[0, 3] = 1 # C11
    W_222[2, 3] = 1 # C21
    
    # M5: (A11+A12)B22
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = -1 # C11
    W_222[1, 4] = 1 # C12
    
    # M6: (A21-A11)(B11+B12)
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[3, 5] = 1 # C22
    
    # M7: (A12-A22)(B21+B22)
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1 # C11
    
    # 2. Construct 4x4x4 Strassen decomposition
    # 7 multiplications of 2x2x2
    # U_444 is 16x49, V_444 is 16x49, W_444 is 16x49
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Strassen combinations for 4x4
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map the 2x2x2 decomposition to these combinations.
    # For each Strassen multiplication Mi, we need to define U, V, W for the 2x2 blocks.
    
    # Helper to add a 2x2x2 decomposition
    def add_222(U_4, V_4, W_4, offset, u_coeff, v_coeff, w_coeff):
        # u_coeff: 4x4 matrix of coefficients for A blocks?
        # No, u_coeff is a vector of 4 values for the 2x2 A block?
        # The 2x2x2 decomposition assumes inputs are 2x2 matrices.
        # Here, the "input" to the 2x2x2 multiplication is a linear combination of A blocks.
        
        # Let's define the linear combinations explicitly.
        # For M1: A_part = A11 + A22, B_part = B11 + B22
        # U for A11: identity, for A22: identity.
        # This is getting complex to generalize with coefficients.
        
        # Let's just hardcode the 7 Strassen steps for 4x4.
        
        pass

    # Alternative: Use the known rank-49 decomposition matrices.
    # Or, simpler: Just use the 114 rank decomposition and verify it.
    
    # Let's build the 114 rank decomposition.
    
    rank_444 = 49
    rank_441 = 16
    total_rank = 2 * rank_444 + rank_441
    
    U_total = np.zeros((16, total_rank))
    V_total = np.zeros((36, total_rank)) # 4*9
    W_total = np.zeros((36, total_rank)) # 9*4
    
    # Part 1: 4x4x4 for cols 0-3
    # We need U_444, V_444, W_444
    # Let's generate them properly.
    
    # Strassen 4x4x4 generation
    # There are 7 "big" multiplications.
    # Each big multiplication is a 2x2x2 tensor product of the block matrices?
    # No, each big multiplication is a scalar multiplication of two 2x2 matrices.
    # So each big multiplication has rank 7 (from 2x2x2).
    # Total 7 * 7 = 49.
    
    # We need the U, V, W for the 7 Strassen combinations.
    
    # Let's define the 7 combinations of A and B blocks.
    # A_blocks: A11, A12, A21, A22 (each 2x2)
    # B_blocks: B11, B12, B21, B22 (each 2x2)
    
    # M1: (A11+A22) * (B11+B22)
    # M2: (A21+A22) * B11
    # M3: A11 * (B12-B22)
    # M4: A22 * (B21-B11)
    # M5: (A11+A12) * B22
    # M6: (A21-A11) * (B11+B12)
    # M7: (A12-A22) * (B21+B22)
    
    # For each Mi, we have A_comb and B_comb.
    # A_comb is a 2x2 matrix formed by linear combos of A_blocks.
    # B_comb is a 2x2 matrix formed by linear combos of B_blocks.
    
    # We can represent A_comb as a 4x4 matrix acting on the flattened A?
    # No, A_comb is a 2x2 matrix where each entry is a linear combo of A entries.
    
    # Let's use the U_222, V_222, W_222 we defined above.
    # U_222 maps A (flattened 2x2) to the 7 scalar products.
    # V_222 maps B (flattened 2x2) to the 7 scalar products.
    # W_222 maps the 7 scalar products to C (flattened 2x2).
    
    # For M1:
    # A_comb = A11 + A22.
    # In the 4x4 A, A11 is rows 0-1, cols 0-1. A22 is rows 2-3, cols 2-3.
    # We need a U matrix that extracts A11 + A22.
    
    # Let's construct U_444 by iterating over the 7 Strassen steps.
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Strassen step definitions
    # Each step has:
    # A_coeffs: 4x4 matrix of coefficients for A blocks?
    # Actually, A_comb is 2x2.
    # A_comb[i,j] = sum_k c[i,j,k] A_block[k]
    
    # Let's define the coefficients for A and B for each of the 7 steps.
    
    # Step 0: M1
    # A11 + A22
    # A_comb[0,0] = A11, A_comb[0,1] = 0, A_comb[1,0] = 0, A_comb[1,1] = A22
    # Wait, A_comb is a 2x2 matrix.
    # M1 = (A11 + A22) * (B11 + B22)
    # This is NOT a standard matrix multiplication of two 2x2 matrices.
    # It is a scalar multiplication of the SUM of two 2x2 matrices?
    # No, Strassen computes $M_1 = (A_{11} + A_{22})(B_{11} + B_{22})$.
    # $A_{11} + A_{22}$ is a 2x2 matrix.
    # So $M_1$ is a 2x2 matrix product.
    # So we use the 2x2x2 decomposition for this product.
    
    # So for each Strassen step, we have a specific A_comb and B_comb.
    
    # Let's define A_comb and B_comb as 4x4 matrices (flattened 2x2 blocks)?
    # No, A_comb is a 2x2 matrix of blocks?
    # No, A_comb is a 2x2 matrix.
    
    # Let's define the linear map from A (4x4) to A_comb (2x2).
    
    # Step 0: M1
    # A_comb = [[A11, 0], [0, A22]]? No.
    # A_comb = A11 + A22?
    # If A_comb is a 2x2 matrix, and A11, A22 are 2x2 matrices.
    # Then A_comb = A11 + A22 is a 2x2 matrix.
    # So A_comb[0,0] = A11[0,0] + A22[0,0], etc.
    
    # We need U such that $U \cdot A_{flat} = A_comb_{flat}$.
    
    # Let's build U_444, V_444, W_444 by composing the 2x2x2 decomposition with the block selection.
    
    # For each of the 7 steps s:
    # 1. Define A_sel (4x4 matrix mapping A_flat to A_comb_flat)
    # 2. Define B_sel (4x4 matrix mapping B_flat to B_comb_flat)
    # 3. The contribution to U_444 is A_sel @ U_222
    # 4. The contribution to V_444 is B_sel @ V_222
    # 5. The contribution to W_444 is W_222 @ C_sel?
    # No, W_222 maps M_flat to C_comb_flat.
    # We need to add C_comb to C.
    # C_comb is a 2x2 matrix.
    # It adds to specific blocks of C.
    
    # Let's define C_add (4x4 matrix mapping C_comb_flat to C_flat)
    
    # Step 0: M1
    # A_comb = A11 + A22
    # A_sel:
    # A_comb_flat = [A11[0,0]+A22[0,0], A11[0,1]+A22[0,1], A11[1,0]+A22[1,0], A11[1,1]+A22[1,1]]
    # A_flat = [A11[0,0], A11[0,1], A11[1,0], A11[1,1], A12[0,0], ...]
    
    # This is tedious but doable.
    
    # Let's write a helper to create the selection matrices.
    
    def make_sel(block_indices, signs):
        # block_indices: list of 4 block indices (0=A11, 1=A12, 2=A21, 3=A22)
        # signs: list of 4 signs
        # Returns 4x4 matrix
        M = np.zeros((4, 16))
        for i, (b, s) in enumerate(zip(block_indices, signs)):
            M[i, b*4:(b+1)*4] = s
        return M
    
    # This is not quite right. A_comb is 2x2.
    # A_comb[0,0] = sum s_k A_k[0,0]
    # A_comb[0,1] = sum s_k A_k[0,1]
    # A_comb[1,0] = sum s_k A_k[1,0]
    # A_comb[1,1] = sum s_k A_k[1,1]
    
    # So A_sel is 4x16.
    # Row 0: coeffs for A_comb[0,0]
    # Row 1: coeffs for A_comb[0,1]
    # Row 2: coeffs for A_comb[1,0]
    # Row 3: coeffs for A_comb[1,1]
    
    # For M1: A_comb = A11 + A22
    # A_comb[0,0] = A11[0,0] + A22[0,0]
    # A_comb[0,1] = A11[0,1] + A22[0,1]
    # A_comb[1,0] = A11[1,0] + A22[1,0]
    # A_comb[1,1] = A11[1,1] + A22[1,1]
    
    # A_sel row 0: 1 at A11[0,0], 1 at A22[0,0]
    # A_sel row 1: 1 at A11[0,1], 1 at A22[0,1]
    # etc.
    
    # Let's define the 7 steps.
    
    steps = [
        # (A_blocks, A_signs, B_blocks, B_signs, C_blocks, C_signs)
        # M1: (A11+A22)(B11+B22) -> C11, C22
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # M2: (A21+A22)B11 -> C21, C22
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # M3: A11(B12-B22) -> C12, C22
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        # M4: A22(B21-B11) -> C11, C21
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        # M5: (A11+A12)B22 -> C11, C12
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # M6: (A21-A11)(B11+B12) -> C22
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # M7: (A12-A22)(B21+B22) -> C11
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    # Note: The C_signs and C_blocks indicate how C_comb adds to C.
    # For M1, C_comb adds to C11 and C22.
    # C_comb is a 2x2 matrix.
    # C_comb[0,0] adds to C11[0,0]?
    # No, M1 is a 2x2 matrix.
    # C11 = M1 + ...
    # So the entire M1 matrix adds to C11 block?
    # No, Strassen formula:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So M1 adds to C11 and C22.
    # M1 is a 2x2 matrix.
    # So M1[0,0] adds to C11[0,0] and C22[0,0]?
    # Yes.
    
    # So C_sel for M1 should map C_comb_flat to C_flat such that:
    # C_comb[0,0] -> C11[0,0] and C22[0,0]
    # C_comb[0,1] -> C11[0,1] and C22[0,1]
    # etc.
    
    # Let's implement this.
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    col_offset = 0
    for a_blocks, a_signs, b_blocks, b_signs, c_blocks, c_signs in steps:
        # Construct A_sel (4x16)
        A_sel = np.zeros((4, 16))
        for k, (b, s) in enumerate(zip(a_blocks, a_signs)):
            # b is block index 0..3
            # We need to add s * A_b to A_comb
            # A_comb is 2x2.
            # A_b is 2x2.
            # So A_comb[i,j] += s * A_b[i,j]
            # A_sel row i, col b*4+j += s
            for i in range(2):
                for j in range(2):
                    A_sel[i, b*4+j] += s
                    A_sel[i+2, b*4+j] += s # Wait, A_comb is 2x2.
                    # A_sel has 4 rows.
                    # Row 0: A_comb[0,0]
                    # Row 1: A_comb[0,1]
                    # Row 2: A_comb[1,0]
                    # Row 3: A_comb[1,1]
                    
        # Correct A_sel construction:
        A_sel = np.zeros((4, 16))
        for b, s in zip(a_blocks, a_signs):
            for i in range(2):
                for j in range(2):
                    A_sel[i, b*4+j] += s
                    A_sel[i+2, b*4+j] += s # No, i+2 is for row 1 of A_comb?
                    # A_comb row 0 is i=0. Row 1 is i=1.
                    # A_sel row 0 is A_comb[0,0].
                    # A_sel row 1 is A_comb[0,1].
                    # A_sel row 2 is A_comb[1,0].
                    # A_sel row 3 is A_comb[1,1].
                    
        # Let's redo A_sel
        A_sel = np.zeros((4, 16))
        for b, s in zip(a_blocks, a_signs):
            # A_b contributes s * A_b to A_comb
            # A_comb[0,0] += s * A_b[0,0]
            # A_comb[0,1] += s * A_b[0,1]
            # A_comb[1,0] += s * A_b[1,0]
            # A_comb[1,1] += s * A_b[1,1]
            A_sel[0, b*4+0] += s
            A_sel[1, b*4+1] += s
            A_sel[2, b*4+2] += s
            A_sel[3, b*4+3] += s
            
        # Similarly for B_sel
        B_sel = np.zeros((4, 16))
        for b, s in zip(b_blocks, b_signs):
            B_sel[0, b*4+0] += s
            B_sel[1, b*4+1] += s
            B_sel[2, b*4+2] += s
            B_sel[3, b*4+3] += s
            
        # C_sel: Maps C_comb_flat (4) to C_flat (16)
        C_sel = np.zeros((16, 4))
        # c_blocks are the blocks of C that receive the sum.
        # c_signs are the signs.
        # C_comb is a 2x2 matrix.
        # C_comb[0,0] adds s * C_comb[0,0] to C_block[0,0]
        for cb, cs in zip(c_blocks, c_signs):
            C_sel[cb*4+0, 0] += cs
            C_sel[cb*4+1, 1] += cs
            C_sel[cb*4+2, 2] += cs
            C_sel[cb*4+3, 3] += cs
            
        # Now combine
        # U_444[:, col_offset:col_offset+7] = A_sel @ U_222
        U_444[:, col_offset:col_offset+7] = A_sel @ U_222
        V_444[:, col_offset:col_offset+7] = B_sel @ V_222
        W_444[:, col_offset:col_offset+7] = C_sel @ W_222
        
        col_offset += 7
        
    # Now U_444, V_444, W_444 are ready for 4x4x4.
    
    # 3. Construct 4x4x1 schoolbook
    U_441 = np.zeros((16, 16))
    V_441 = np.zeros((4, 16)) # 4*1
    W_441 = np.zeros((4, 16)) # 1*4
    
    for i in range(4):
        for j in range(4):
            r = i*4 + j
            U_441[i*4+j, r] = 1
            V_441[j, r] = 1
            W_441[i, r] = 1 # k=0, so W[0*4+i] = W[i]
            
    # 4. Assemble total
    # B is 4x9.
    # B1 is 4x4 (cols 0-3)
    # B2 is 4x4 (cols 4-7)
    # B3 is 4x1 (col 8)
    
    U_total = np.zeros((16, 114))
    V_total = np.zeros((36, 114))
    W_total = np.zeros((36, 114))
    
    # Part 1: B1
    U_total[:, 0:49] = U_444
    # V_444 is 16x49. V_total is 36x114.
    # V_444 corresponds to B1 (cols 0-3).
    # V_total rows 0-15 correspond to B1.
    V_total[0:16, 0:49] = V_444
    # W_444 is 16x49. W_total is 36x114.
    # W_444 corresponds to C1 (cols 0-3).
    # W_total rows 0-15 correspond to C1.
    W_total[0:16, 0:49] = W_444
    
    # Part 2: B2
    U_total[:, 49:98] = U_444
    # V_444 corresponds to B2 (cols 4-7).
    # V_total rows 16-31 correspond to B2.
    V_total[16:32, 49:98] = V_444
    # W_444 corresponds to C2 (cols 4-7).
    # W_total rows 16-31 correspond to C2.
    W_total[16:32, 49:98] = W_444
    
    # Part 3: B3
    U_total[:, 98:114] = U_441
    # V_441 corresponds to B3 (col 8).
    # V_total rows 32-35 correspond to B3.
    V_total[32:36, 98:114] = V_441
    # W_441 corresponds to C3 (col 8).
    # W_total rows 32-35 correspond to C3.
    W_total[32:36, 98:114] = W_441
    
    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# Verify the logic
# U_444 is 16x49.
# V_444 is 16x49.
# W_444 is 16x49.
# U_441 is 16x16.
# V_441 is 4x16.
# W_441 is 4x16.

# Total U: 16x114.
# Total V: 36x114.
# Total W: 36x114.

# The shapes match the interface.
# U: (n*m, rank) = (16, 114)
# V: (m*p, rank) = (36, 114)
# W: (p*n, rank) = (36, 114)

# One detail: W_441 indexing.
# W[k*n+i, r].
# For B3, k=8.
# W_total[32:36, 98:114] corresponds to k=8.
# W_total[8*4+i, r] = W_total[32+i, r].
# My W_441 has rows 0-3.
# So W_total[32:36, 98:114] = W_441 is correct.

# V_441 indexing.
# V[j*p+k, r].
# For B3, k=8.
# V_total[j*9+8, r] = V_total[j*9+8, 98:114].
# My V_441 has rows 0-3 (j=0..3).
# V_total[32:36, 98:114] corresponds to j=0..3, k=8.
# j*9+8:
# j=0 -> 8
# j=1 -> 17
# j=2 -> 26
# j=3 -> 35
# These are NOT contiguous in V_total.
# V_total is 36x114.
# Rows are j*9+k.
# For k=8, rows are 8, 17, 26, 35.
# So I cannot just assign a contiguous block.

# I need to scatter V_441 into V_total.

# Let's fix the assembly.

