import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct the rank-98 decomposition.
    # It consists of two independent rank-49 decompositions for 4x4x4.
    # One for columns 0-3 of B, one for columns 4-7 of B.
    
    def get_strassen_444_factors():
        # Returns U, V, W for 4x4x4 with rank 49.
        # U shape (16, 49), V shape (16, 49), W shape (16, 49)
        
        U = np.zeros((16, 49), dtype=float)
        V = np.zeros((16, 49), dtype=float)
        W = np.zeros((16, 49), dtype=float)
        
        # Strassen steps for 4x4 (block level)
        # Each step r (0..6) involves adding/subtracting 2x2 blocks.
        # Let's define the block indices for A and B for each step.
        # A blocks: A00, A01, A10, A11
        # B blocks: B00, B01, B10, B11
        
        # Step 0: (A00+A11)(B00+B11)
        # Step 1: (A10+A11)B00
        # Step 2: A00(B01-B11)
        # Step 3: A11(B10-B00)
        # Step 4: (A00+A01)B11
        # Step 5: (A10-A00)(B00+B01)
        # Step 6: (A01-A11)(B10+B11)
        
        # For each step, we need to know which scalar indices of A and B are used.
        # The blocks are 2x2 matrices.
        # A00: indices 0,1,4,5 (row-major: 00,01,10,11 -> 0*4+0, 0*4+1, 1*4+0, 1*4+1)
        # Wait, index mapping:
        # A[i,j] is at index i*m+j = i*4+j.
        # Block A00 (top-left 2x2): i in 0,1; j in 0,1. Indices: 0,1,4,5.
        # Block A01 (top-right 2x2): i in 0,1; j in 2,3. Indices: 2,3,6,7.
        # Block A10 (bottom-left 2x2): i in 2,3; j in 0,1. Indices: 8,9,12,13.
        # Block A11 (bottom-right 2x2): i in 2,3; j in 2,3. Indices: 10,11,14,15.
        
        block_indices = {
            '00': [0, 1, 4, 5],
            '01': [2, 3, 6, 7],
            '10': [8, 9, 12, 13],
            '11': [10, 11, 14, 15]
        }
        
        # Strassen scalar decomposition for 2x2x2 (rank 7)
        # This operates on the entries of the 2x2 matrices X and Y.
        # X = [[x00, x01], [x10, x11]]
        # Y = [[y00, y01], [y10, y11]]
        # The 7 multiplications are:
        # m1 = (x00+x11)(y00+y11)
        # m2 = (x10+x11)y00
        # m3 = x00(y01-y11)
        # m4 = x11(y10-y00)
        # m5 = (x00+x01)y11
        # m6 = (x10-x00)(y00+y01)
        # m7 = (x01-x11)(y10+y11)
        
        # We need to map the local indices (00, 01, 10, 11) of the 2x2 block
        # to the global indices of A and B.
        
        # For a given step r, we define X_r and Y_r as linear combinations of blocks.
        # Let's define the coefficients for each block in X_r and Y_r.
        
        # Step definitions: (coeffs for A blocks, coeffs for B blocks)
        # A blocks order: 00, 01, 10, 11
        # B blocks order: 00, 01, 10, 11
        
        steps = [
            # r=0
            ([1, 0, 0, 1], [1, 0, 0, 1]),  # (A00+A11)(B00+B11)
            # r=1
            ([0, 0, 1, 1], [1, 0, 0, 0]),  # (A10+A11)B00
            # r=2
            ([1, 0, 0, 0], [0, 1, 0, -1]), # A00(B01-B11)
            # r=3
            ([0, 0, 0, 1], [0, 0, 1, -1]), # A11(B10-B00)
            # r=4
            ([1, 1, 0, 0], [0, 0, 0, 1]),  # (A00+A01)B11
            # r=5
            ([0, 0, 1, -1], [1, 1, 0, 0]), # (A10-A00)(B00+B01)
            # r=6
            ([0, 1, 0, -1], [0, 0, 1, 1])  # (A01-A11)(B10+B11)
        ]
        
        # Output combination for C blocks
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M_r is the result of step r (a 2x2 matrix of blocks? No, M_r is a 2x2 matrix of scalars? 
        # No, in the 4x4 context, M_r is a 2x2 matrix of 2x2 blocks? 
        # No. Let's re-evaluate.
        
        # In the 4x4 Strassen:
        # We compute 7 products M_r of 2x2 matrices (which are blocks of A and B).
        # Each M_r is a 2x2 matrix.
        # The entries of M_r are computed using the 2x2 Strassen algorithm.
        # So M_r has 4 entries, each computed by summing 7 scalar products.
        # Total 7 steps * 4 entries * 7 scalars/entry? No.
        # The rank is the number of scalar multiplications.
        # Each M_r is a 2x2x2 tensor multiplication. Rank 7.
        # So each M_r requires 7 scalar multiplications.
        # Total 7 * 7 = 49 scalar multiplications.
        
        # The output C is a 4x4 matrix.
        # C is composed of 4 blocks C00, C01, C10, C11, each 2x2.
        # The blocks C_xy are linear combinations of the blocks M_r?
        # No, C_xy are linear combinations of the *entries* of M_r?
        # Actually, the Strassen formula for 4x4 says:
        # C00 = M1 + M4 - M5 + M7
        # where M1, M4, M5, M7 are 2x2 matrices (the results of the block multiplications).
        # So each entry of C00 is a sum of corresponding entries of M1, M4, M5, M7.
        
        # So, for each scalar multiplication t (0..48), we need to determine:
        # 1. Which step r it belongs to (r = t // 7).
        # 2. Which scalar multiplication s within that step (s = t % 7).
        # 3. Which scalar indices of A and B are involved in that scalar multiplication.
        # 4. Which scalar index of C receives the result, and with what sign.
        
        # Let's iterate through steps r and scalar subssteps s.
        
        # Strassen 2x2 scalar factors:
        # s=0: (x00+x11)(y00+y11) -> contributes to z00, z11
        # s=1: (x10+x11)y00 -> contributes to z10, z11
        # s=2: x00(y01-y11) -> contributes to z01, z11
        # s=3: x11(y10-y00) -> contributes to z10, z00
        # s=4: (x00+x01)y11 -> contributes to z01, z11
        # s=5: (x10-x00)(y00+y01) -> contributes to z10, z11
        # s=6: (x01-x11)(y10+y11) -> contributes to z01, z10
        
        # Wait, the standard Strassen output is:
        # z00 = m1 + m4 - m5 + m7
        # z01 = m3 + m5
        # z10 = m2 + m4
        # z11 = m1 - m2 + m3 + m6
        
        # Let's define the contribution of each scalar multiplication m_s to the entries of the 2x2 result matrix Z.
        # m_s is a scalar.
        # We need to know which entry of Z (00, 01, 10, 11) it adds to, and with what coefficient.
        
        strassen_222_contributions = [
            # s, z00, z01, z10, z11
            (1, 0, 0, 1),   # m1
            (0, 0, 1, -1),  # m2
            (0, 1, 0, 1),   # m3
            (1, 0, 1, 0),   # m4
            (-1, 1, 0, 0),  # m5
            (0, 0, 0, 1),   # m6
            (1, 0, 0, 0)    # m7
        ]
        # Note: The coefficients above are for the entries of Z.
        # m1 contributes +1 to z00, +1 to z11.
        # m2 contributes +1 to z10, -1 to z11.
        # m3 contributes +1 to z01, +1 to z11.
        # m4 contributes +1 to z00, +1 to z10.
        # m5 contributes -1 to z00, +1 to z01.
        # m6 contributes +1 to z11.
        # m7 contributes +1 to z00.
        
        # Now, for each step r (0..6), we have X_r and Y_r.
        # X_r is a 2x2 matrix of blocks.
        # Y_r is a 2x2 matrix of blocks.
        # The scalar multiplication m_s within step r involves specific scalar entries of X_r and Y_r.
        # We need to map the local indices of X_r and Y_r (00, 01, 10, 11) to global indices of A and B.
        
        # The local indices of X_r correspond to the blocks of A.
        # X_r[0,0] is a linear combination of A blocks.
        # Specifically, X_r[0,0] = sum(c_ik * A_ik) where c_ik are from the step definition.
        # But X_r itself is a 2x2 matrix.
        # X_r[0,0] is the top-left 2x2 block of the "super-matrix" X_r.
        # This block is formed by the linear combination of the *corresponding blocks* of A.
        
        # Let's clarify the structure.
        # A is 4x4.
        # A_00 is the top-left 2x2 block of A.
        # In step r, we form X_r.
        # X_r is a 2x2 matrix where each entry is a 2x2 matrix.
        # X_r[0,0] = c_A_00 * A_00 + c_A_01 * A_01 + c_A_10 * A_10 + c_A_11 * A_11?
        # No. The Strassen algorithm for 4x4 operates on the 2x2 blocks.
        # The "matrices" in the Strassen formula are the 2x2 blocks themselves.
        # So X_r is a 2x2 matrix of *blocks*.
        # X_r[0,0] is a 2x2 matrix (a block).
        # X_r[0,0] = sum of blocks from A.
        
        # Let's define the block coefficients for each step.
        # Step r defines how to form the 2x2 matrix of blocks X_r and Y_r.
        # X_r is a 2x2 matrix of 2x2 blocks.
        # So X_r has 4 entries, each a 2x2 matrix.
        # X_r[0,0] = c00 * A_00 + c01 * A_01 + c10 * A_10 + c11 * A_11 ?
        # No, the Strassen formula for 4x4 is:
        # M1 = (A00+A11)(B00+B11)
        # Here A00, A11, B00, B11 are 2x2 blocks.
        # The product M1 is a 2x2 matrix of blocks? No, M1 is a 2x2 matrix (since it's the product of two 2x2 matrices).
        # Wait. A00 is 2x2. B00 is 2x2. Their product is 2x2.
        # So M1 is a 2x2 matrix.
        # But M1 is one of the 7 "meta" products.
        # And M1 is used to form C00, C01, etc.
        # C00 is a 2x2 block.
        # C00 = M1 + M4 - M5 + M7.
        # So C00 (2x2) is the sum of M1, M4, M5, M7 (each 2x2).
        
        # So, for each step r, we compute a 2x2 matrix M_r.
        # M_r = X_r * Y_r, where X_r and Y_r are 2x2 matrices formed from the blocks of A and B.
        # To compute M_r (2x2 matrix), we use the rank-7 Strassen decomposition for 2x2x2.
        # This decomposition computes the 4 entries of M_r using 7 scalar multiplications.
        
        # So, for step r:
        # 1. Determine X_r and Y_r.
        #    X_r is a 2x2 matrix. Its entries are linear combinations of the *blocks* of A.
        #    Wait, X_r[0,0] is a scalar? No.
        #    In the 2x2 Strassen, the inputs are 2x2 matrices.
        #    Here, the inputs to the "inner" Strassen are the blocks?
        #    No. The "outer" Strassen reduces the 4x4 problem to 7 products of 2x2 matrices.
        #    The "inner" Strassen reduces each 2x2 product to 7 scalar products.
        
        # So, for step r, we have specific 2x2 matrices X_r and Y_r.
        # X_r is formed by adding/subtracting the 2x2 blocks of A.
        # Y_r is formed by adding/subtracting the 2x2 blocks of B.
        
        # Let's define the block coefficients for X_r and Y_r.
        # X_r[0,0], X_r[0,1], X_r[1,0], X_r[1,1] are 2x2 matrices.
        # Each is a linear combination of A_00, A_01, A_10, A_11.
        
        # Step 0: X = [[A00, 0], [0, A11]]? No.
        # M1 = (A00+A11)(B00+B11).
        # This is a product of two 2x2 matrices.
        # Let U1 = A00+A11 (2x2 matrix).
        # Let V1 = B00+B11 (2x2 matrix).
        # M1 = U1 V1.
        
        # So for step 0:
        # U1 = A00 + A11.
        # V1 = B00 + B11.
        
        # U1 is a 2x2 matrix.
        # U1[0,0] = A00[0,0] + A11[0,0]
        # U1[0,1] = A00[0,1] + A11[0,1]
        # ...
        
        # So, to get the scalar factors for U1 and V1, we need to know which scalar indices of A and B contribute to each entry of U1 and V1.
        
        # Let's define the block coefficients for each step.
        # For step r, we have 4 "positions" in the 2x2 matrix U_r (top-left, top-right, bottom-left, bottom-right).
        # Each position is a linear combination of the 4 blocks of A.
        
        # Step 0:
        # U_r is 2x2.
        # U_r[0,0] = A00 + A11 ? No.
        # The formula is M1 = (A00+A11)(B00+B11).
        # This implies that the "left factor" is the 2x2 matrix (A00+A11).
        # But A00+A11 is a 2x2 matrix.
        # So the left factor for the inner Strassen is the 2x2 matrix A00+A11.
        # The right factor is the 2x2 matrix B00+B11.
        
        # So, for step r, we define:
        # Left_r: a 2x2 matrix, where each entry is a linear combination of A blocks.
        # Right_r: a 2x2 matrix, where each entry is a linear combination of B blocks.
        
        # Step 0:
        # Left_r = [[A00+A11, 0], [0, 0]]? No.
        # The Strassen formula for 4x4 is:
        # M1 = (A00+A11)(B00+B11)
        # This is a single 2x2 matrix product.
        # So Left_r is the 2x2 matrix A00+A11.
        # Right_r is the 2x2 matrix B00+B11.
        
        # Wait, A00+A11 is a 2x2 matrix.
        # So Left_r is just that matrix.
        # It doesn't have a 2x2 structure of blocks?
        # Yes it does. Left_r is a 2x2 matrix of scalars? No, of 2x2 blocks?
        # No. A00 is a 2x2 matrix of scalars.
        # So A00+A11 is a 2x2 matrix of scalars.
        # So Left_r is a 2x2 matrix of scalars.
        
        # So the "inner" Strassen operates on these 2x2 matrices of scalars.
        
        # So, for step r:
        # Left_r is a 2x2 matrix.
        # Left_r[i,j] = sum( c_ik * A_k[i,j] ) where k is the block index.
        
        # Let's define the coefficients for each step.
        # The blocks are A00, A01, A10, A11.
        
        # Step 0:
        # Left = A00 + A11.
        # So Left[0,0] = A00[0,0] + A11[0,0]
        # Left[0,1] = A00[0,1] + A11[0,1]
        # Left[1,0] = A00[1,0] + A11[1,0]
        # Left[1,1] = A00[1,1] + A11[1,1]
        
        # Right = B00 + B11.
        
        # Step 1:
        # M2 = (A10+A11)B00.
        # Left = A10 + A11.
        # Right = B00.
        
        # Step 2:
        # M3 = A00(B01-B11).
        # Left = A00.
        # Right = B01 - B11.
        
        # Step 3:
        # M4 = A11(B10-B00).
        # Left = A11.
        # Right = B10 - B00.
        
        # Step 4:
        # M5 = (A00+A01)B11.
        # Left = A00 + A01.
        # Right = B11.
        
        # Step 5:
        # M6 = (A10-A00)(B00+B01).
        # Left = A10 - A00.
        # Right = B00 + B01.
        
        # Step 6:
        # M7 = (A01-A11)(B10+B11).
        # Left = A01 - A11.
        # Right = B10 + B11.
        
        # Now we can construct the factors.
        
        block_coeffs = [
            # (Left coeffs for A00, A01, A10, A11), (Right coeffs for B00, B01, B10, B11)
            ([1, 0, 0, 1], [1, 0, 0, 1]), # Step 0
            ([0, 0, 1, 1], [1, 0, 0, 0]), # Step 1
            ([1, 0, 0, 0], [0, 1, 0, -1]),# Step 2
            ([0, 0, 0, 1], [0, 0, 1, -1]),# Step 3
            ([1, 1, 0, 0], [0, 0, 0, 1]), # Step 4
            ([0, 0, 1, -1], [1, 1, 0, 0]),# Step 5
            ([0, 1, 0, -1], [0, 0, 1, 1]) # Step 6
        ]
        
        # Strassen 2x2 scalar factors
        # s, (x00, x01, x10, x11), (y00, y01, y10, y11), (z00, z01, z10, z11)
        # We need the coefficients for x and y for each scalar multiplication.
        
        strassen_222_factors = [
            # s, x_coeffs, y_coeffs
            (0, [1, 0, 0, 1], [1, 0, 0, 1]), # m1 = (x00+x11)(y00+y11)
            (1, [0, 0, 1, 1], [1, 0, 0, 0]), # m2 = (x10+x11)y00
            (2, [1, 0, 0, 0], [0, 1, 0, -1]),# m3 = x00(y01-y11)
            (3, [0, 0, 0, 1], [0, 0, 1, -1]),# m4 = x11(y10-y00)
            (4, [1, 1, 0, 0], [0, 0, 0, 1]), # m5 = (x00+x01)y11
            (5, [0, 0, 1, -1], [1, 1, 0, 0]),# m6 = (x10-x00)(y00+y01)
            (6, [0, 1, 0, -1], [0, 0, 1, 1]) # m7 = (x01-x11)(y10+y11)
        ]
        
        # Output contributions for Z (2x2 matrix)
        # z00, z01, z10, z11
        strassen_222_output = [
            (1, 0, 0, 1),   # m1
            (0, 0, 1, -1),  # m2
            (0, 1, 0, 1),   # m3
            (1, 0, 1, 0),   # m4
            (-1, 1, 0, 0),  # m5
            (0, 0, 0, 1),   # m6
            (1, 0, 0, 0)    # m7
        ]
        
        # Now, for each step r and each scalar multiplication s:
        # 1. Get x_coeffs and y_coeffs from strassen_222_factors[s].
        # 2. Get block_coeffs for step r.
        # 3. The scalar indices of A involved are determined by:
        #    x_coeffs[i] * block_coeffs[r][i] for each block i.
        #    For each block i with non-zero coefficient, we add the corresponding scalar indices.
        
        # Let's iterate.
        
        for r in range(7):
            a_block_coeffs, b_block_coeffs = block_coeffs[r]
            
            for s in range(7):
                t = r * 7 + s
                
                x_coeffs, y_coeffs = strassen_222_factors[s]
                
                # Determine A indices
                for i in range(4):
                    if x_coeffs[i] != 0 and a_block_coeffs[i] != 0:
                        coeff = x_coeffs[i] * a_block_coeffs[i]
                        if coeff != 0:
                            for idx in block_indices['00' if i==0 else '01' if i==1 else '10' if i==2 else '11']:
                                U[idx, t] += coeff
                                
                # Determine B indices
                for i in range(4):
                    if y_coeffs[i] != 0 and b_block_coeffs[i] != 0:
                        coeff = y_coeffs[i] * b_block_coeffs[i]
                        if coeff != 0:
                            for idx in block_indices['00' if i==0 else '01' if i==1 else '10' if i==2 else '11']:
                                V[idx, t] += coeff
                                
                # Determine C indices
                # The result of this scalar multiplication contributes to the entries of M_r.
                # M_r is a 2x2 matrix.
                # The entries of M_r contribute to the blocks of C.
                
                # M_r[0,0] contributes to C00
                # M_r[0,1] contributes to C01
                # M_r[1,0] contributes to C10
                # M_r[1,1] contributes to C11
                
                # The contribution of scalar multiplication s to M_r's entries is given by strassen_222_output[s].
                z00, z01, z10, z11 = strassen_222_output[s]
                
                # We need to map M_r's entries to C's blocks.
                # The Strassen 4x4 output combination is:
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                
                # So, for step r, the contribution to C blocks is:
                # If r==0 (M1): C00 += 1*M1, C11 += 1*M1
                # If r==1 (M2): C10 += 1*M2, C11 -= 1*M2
                # If r==2 (M3): C01 += 1*M3, C11 += 1*M3
                # If r==3 (M4): C00 += 1*M4, C10 += 1*M4
                # If r==4 (M5): C00 -= 1*M5, C01 += 1*M5
                # If r==5 (M6): C11 += 1*M6
                # If r==6 (M7): C00 += 1*M7
                
                # Let's define the block output coefficients for each step.
                # c00, c01, c10, c11
                step_output_coeffs = [
                    (1, 0, 0, 1),   # r=0 (M1)
                    (0, 0, 1, -1),  # r=1 (M2)
                    (0, 1, 0, 1),   # r=2 (M3)
                    (1, 0, 1, 0),   # r=3 (M4)
                    (-1, 1, 0, 0),  # r=4 (M5)
                    (0, 0, 0, 1),   # r=5 (M6)
                    (1, 0, 0, 0)    # r=6 (M7)
                ]
                
                c00, c01, c10, c11 = step_output_coeffs[r]
                
                # Now, for each entry of M_r (z00, z01, z10, z11), we add to the corresponding block of C.
                # M_r[0,0] is z00. It contributes to C00 with coefficient c00.
                # So C00[0,0] += z00 * c00
                # C00[0,1] += z00 * c00
                # C00[1,0] += z00 * c00
                # C00[1,1] += z00 * c00
                
                # Wait, M_r is a 2x2 matrix.
                # M_r[0,0] is a scalar.
                # C00 is a 2x2 matrix.
                # C00 = c00 * M_r + ...
                # So each entry of C00 is c00 * M_r[0,0] + ...?
                # No. C00 is a 2x2 matrix. M_r is a 2x2 matrix.
                # C00[0,0] = c00 * M_r[0,0] + ...
                # C00[0,1] = c00 * M_r[0,1] + ...
                
                # So, for the scalar multiplication t, which computes a scalar value val = U[..] * V[..]:
                # This val is added to M_r[z_idx] with coefficient z_coeff.
                # Then M_r[z_idx] is added to C_block[block_idx] with coefficient c_coeff.
                # So val is added to C_block[block_idx][z_idx] with coefficient z_coeff * c_coeff.
                
                # Let's map z_idx to C block indices.
                # z_idx 0 -> M_r[0,0] -> contributes to C00[0,0], C00[0,1]? No.
                # M_r[0,0] is the top-left entry of the 2x2 matrix M_r.
                # C00 is the top-left 2x2 block of C.
                # C00[0,0] corresponds to M_r[0,0].
                # C00[0,1] corresponds to M_r[0,1].
                # C00[1,0] corresponds to M_r[1,0].
                # C00[1,1] corresponds to M_r[1,1].
                
                # So, for z_idx 0 (M_r[0,0]):
                # It contributes to C00[0,0] with coeff c00.
                # For z_idx 1 (M_r[0,1]):
                # It contributes to C00[0,1] with coeff c00.
                # For z_idx 2 (M_r[1,0]):
                # It contributes to C00[1,0] with coeff c00.
                # For z_idx 3 (M_r[1,1]):
                # It contributes to C00[1,1] with coeff c00.
                
                # So, for each z_idx (0..3):
                # z_coeff = strassen_222_output[s][z_idx]
                # c_block_idx = z_idx // 2? No.
                # z_idx 0 -> C00
                # z_idx 1 -> C01
                # z_idx 2 -> C10
                # z_idx 3 -> C11
                
                # Wait, M_r is a 2x2 matrix.
                # M_r[0,0] is z00.
                # M_r[0,1] is z01.
                # M_r[1,0] is z10.
                # M_r[1,1] is z11.
                
                # C00 gets M_r[0,0]? No.
                # C00 = c00 * M_r.
                # So C00[0,0] = c00 * M_r[0,0]
                # C00[0,1] = c00 * M_r[0,1]
                # C00[1,0] = c00 * M_r[1,0]
                # C00[1,1] = c00 * M_r[1,1]
                
                # So, for a given z_idx (0..3), which C block does it contribute to?
                # It contributes to ALL blocks? No.
                # The step r produces ONE matrix M_r.
                # This M_r is added to C00, C01, C10, C11 with different coefficients.
                # So M_r[0,0] contributes to C00[0,0], C01[0,0], C10[0,0], C11[0,0]?
                # No.
                # C00 = c00 * M_r.
                # C01 = c01 * M_r.
                # C10 = c10 * M_r.
                # C11 = c11 * M_r.
                
                # So M_r[0,0] contributes to:
                # C00[0,0] with c00
                # C01[0,0] with c01
                # C10[0,0] with c10
                # C11[0,0] with c11
                
                # M_r[0,1] contributes to:
                # C00[0,1] with c00
                # C01[0,1] with c01
                # ...
                
                # So, for each z_idx (0..3):
                # z_coeff = strassen_222_output[s][z_idx]
                # If z_coeff != 0:
                #   For each block b (0..3):
                #     c_coeff = step_output_coeffs[r][b]
                #     If c_coeff != 0:
                #       total_coeff = z_coeff * c_coeff
                #       If total_coeff != 0:
                #         # Add to C block b, at position z_idx
                #         # C block b is 2x2.
                #         # z_idx 0 -> (0,0)
                #         # z_idx 1 -> (0,1)
                #         # z_idx 2 -> (1,0)
                #         # z_idx 3 -> (1,1)
                #         # Global index in W:
                #         # W index = k*n + i
                #         # Block 0 (C00): k in 0,1; i in 0,1.
                #         #   (0,0) -> k=0, i=0 -> 0*4+0 = 0
                #         #   (0,1) -> k=0, i=1 -> 0*4+1 = 1
                #         #   (1,0) -> k=1, i=0 -> 1*4+0 = 4
                #         #   (1,1) -> k=1, i=1 -> 1*4+1 = 5
                #         # Block 1 (C01): k in 0,1; i in 2,3.
                #         #   (0,0) -> k=0, i=2 -> 0*4+2 = 2
                #         #   (0,1) -> k=0, i=3 -> 0*4+3 = 3
                #         #   (1,0) -> k=1, i=2 -> 1*4+2 = 6
                #         #   (1,1) -> k=1, i=3 -> 1*4+3 = 7
                #         # Block 2 (C10): k in 2,3; i in 0,1.
                #         #   (0,0) -> k=2, i=0 -> 2*4+0 = 8
                #         #   (0,1) -> k=2, i=1 -> 2*4+1 = 9
                #         #   (1,0) -> k=3, i=0 -> 3*4+0 = 12
                #         #   (1,1) -> k=3, i=1 -> 3*4+1 = 13
                #         # Block 3 (C11): k in 2,3; i in 2,3.
                #         #   (0,0) -> k=2, i=2 -> 2*4+2 = 10
                #         #   (0,1) -> k=2, i=3 -> 2*4+3 = 11
                #         #   (1,0) -> k=3, i=2 -> 3*4+2 = 14
                #         #   (1,1) -> k=3, i=3 -> 3*4+3 = 15
                
                # Let's define the W indices for each block and z_idx.
                w_indices = [
                    [0, 1, 4, 5],   # Block 0
                    [2, 3, 6, 7],   # Block 1
                    [8, 9, 12, 13], # Block 2
                    [10, 11, 14, 15]# Block 3
                ]
                
                for z_idx in range(4):
                    z_coeff = strassen_222_output[s][z_idx]
                    if z_coeff == 0: continue
                    for b in range(4):
                        c_coeff = step_output_coeffs[r][b]
                        if c_coeff == 0: continue
                        total_coeff = z_coeff * c_coeff
                        if total_coeff == 0: continue
                        w_idx = w_indices[b][z_idx]
                        W[w_idx, t] += total_coeff
                        
        return U, V, W

    # Get factors for 4x4x4
    U_444, V_444, W_444 = get_strassen_444_factors()
    
    # Construct 4x4x8 factors
    # Two copies, one for p=0..3, one for p=4..7
    # U shape (16, 98)
    # V shape (32, 98)
    # W shape (32, 98)
    
    U = np.zeros((16, 98), dtype=float)
    V = np.zeros((32, 98), dtype=float)
    W = np.zeros((32, 98), dtype=float)
    
    # First half: p=0..3
    U[:, 0:49] = U_444
    V[0:16, 0:49] = V_444
    W[0:16, 0:49] = W_444
    
    # Second half: p=4..7
    U[:, 49:98] = U_444
    V[16:32, 49:98] = V_444
    W[16:32, 49:98] = W_444
    
    return {"U": U, "V": V, "W": W}
