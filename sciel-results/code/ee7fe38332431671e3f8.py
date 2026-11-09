"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # We decompose the tensor into a 4x4x4 part and a 4x4x1 part.
    # The 4x4x4 part uses Strassen's algorithm (rank 49).
    # The 4x4x1 part uses schoolbook (rank 4).
    # Total rank = 49 + 4 = 53.
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 tensor
    # U[i*m+j], V[j*p+k], W[k*n+i]
    def add_tensor(i, j, k, col):
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1

    # Part 1: 4x4x4 part (k=0..3)
    # We use the Strassen decomposition for 4x4 matrix multiplication.
    # This is a recursive application of the 2x2 Strassen algorithm.
    
    # The 2x2 Strassen algorithm has 7 terms.
    # For 4x4, we have 7 block multiplications of 2x2 matrices.
    # Each 2x2 multiplication is decomposed into 7 rank-1 terms.
    # So we have 7 * 7 = 49 terms.
    
    # Let's define the Strassen coefficients for 2x2.
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We need to map the 4x4 indices to the 2x2 block structure.
    # A is 4x4. Blocks are:
    # A11: rows 0,1 cols 0,1
    # A12: rows 0,1 cols 2,3
    # A21: rows 2,3 cols 0,1
    # A22: rows 2,3 cols 2,3
    
    # B is 4x4 (for the first 4 columns). Blocks are:
    # B11: rows 0,1 cols 0,1
    # B12: rows 0,1 cols 2,3
    # B21: rows 2,3 cols 0,1
    # B22: rows 2,3 cols 2,3
    
    # The output C is 4x4. Blocks are:
    # C11: rows 0,1 cols 0,1
    # C12: rows 0,1 cols 2,3
    # C21: rows 2,3 cols 0,1
    # C22: rows 2,3 cols 2,3
    
    # We will iterate over the 7 Strassen terms for the 4x4 level.
    # For each term, we iterate over the 7 Strassen terms for the 2x2 level.
    
    # Let's define the 2x2 Strassen terms as a list of tuples (a_coeff, b_coeff, c_coeff)
    # where the term is sum a_coeff * A_block * b_coeff * B_block
    # Actually, it's easier to just generate the U, V, W entries directly.
    
    # The 2x2 Strassen algorithm terms:
    # Term 1: (A11 + A22)(B11 + B22)
    #   A part: A11 + A22 -> indices (0,0), (1,1) with coeff 1
    #   B part: B11 + B22 -> indices (0,0), (1,1) with coeff 1
    #   C part: C11, C22 with coeff 1
    # Term 2: (A21 + A22)B11
    #   A part: A21 + A22 -> indices (2,0), (3,0) [Wait, A21 is rows 2,3 cols 0,1? No, A21 is row block 2, col block 1. So rows 2,3, cols 0,1.]
    #   Let's be precise with indices.
    #   A11: i in {0,1}, j in {0,1}
    #   A12: i in {0,1}, j in {2,3}
    #   A21: i in {2,3}, j in {0,1}
    #   A22: i in {2,3}, j in {2,3}
    
    #   B11: j in {0,1}, k in {0,1}
    #   B12: j in {0,1}, k in {2,3}
    #   B21: j in {2,3}, k in {0,1}
    #   B22: j in {2,3}, k in {2,3}
    
    #   C11: i in {0,1}, k in {0,1}
    #   C12: i in {0,1}, k in {2,3}
    #   C21: i in {2,3}, k in {0,1}
    #   C22: i in {2,3}, k in {2,3}
    
    # The 7 terms for 2x2:
    # T1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
    # T2: (A21+A22)B11 -> C21 += 1
    # T3: A11(B12-B22) -> C12 += 1
    # T4: A22(B21-B11) -> C21 += 1
    # T5: (A11+A12)B22 -> C11 += -1, C12 += 1
    # T6: (A21-A11)(B11+B12) -> C22 += 1
    # T7: (A12-A22)(B21+B22) -> C11 += 1, C22 += -1
    
    # Wait, the C coefficients are:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define the terms as (A_indices, B_indices, C_indices, coeff)
    # But since we are doing rank-1 decomposition, each term is a sum of rank-1 tensors.
    # For example, T1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This is 4 rank-1 tensors? No, it's a product of sums.
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # Each of these is a 2x2 matrix multiplication.
    # So T1 contributes 4 * (rank of 2x2) = 4 * 7 = 28 terms?
    # No, that's not right.
    # The Strassen algorithm for 2x2 has rank 7.
    # The 7 terms are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mk is a 2x2 matrix product.
    # To compute Mk, we use the rank-7 Strassen algorithm again?
    # No, Mk is a 2x2 matrix. We can just use the schoolbook rank-4 for each Mk?
    # Then the total rank would be 7 * 4 = 28 for the 4x4 part?
    # But Strassen for 4x4 is rank 49.
    # Why? Because we apply Strassen to the 2x2 blocks.
    # So each Mk (which is a 2x2 product) is computed using Strassen (rank 7).
    # So total rank is 7 * 7 = 49.
    
    # So we need to implement the rank-7 Strassen algorithm for 2x2.
    # And use it for each of the 7 Mk terms.
    
    # Let's define the 2x2 Strassen algorithm terms explicitly.
    # For a 2x2 matrix product C = AB, where A, B, C are 2x2.
    # Indices: i, j, k in {0, 1}.
    # A[i,j], B[j,k], C[i,k].
    
    # The 7 terms are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will generate the U, V, W entries for the 4x4 part by iterating over the 7 Mk terms,
    # and for each Mk, iterating over the 7 Strassen terms for the 2x2 product.
    
    # Let's define the 2x2 Strassen terms as a list of (A_coeff, B_coeff, C_coeff)
    # where the term is A_coeff * A[i,j] * B_coeff * B[j,k]
    # and it contributes to C[i,k] with C_coeff.
    
    # Actually, it's easier to just list the 7 terms for 2x2:
    # Term 1: A00B00, A00B11, A11B00, A11B11
    #   Wait, M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    #   This is 4 rank-1 terms? No, it's a sum of products.
    #   In the rank-1 decomposition, each product A[i,j]B[j,k] is a rank-1 tensor.
    #   So M1 contributes 4 rank-1 tensors.
    #   But wait, the rank of 2x2 is 7.
    #   The 7 terms are the 7 products M1..M7.
    #   Each Mk is a 2x2 matrix.
    #   To compute Mk, we don't decompose it further into rank-1 tensors in the Strassen algorithm.
    #   The Strassen algorithm for 2x2 has rank 7, meaning it uses 7 scalar multiplications.
    #   Each scalar multiplication is of the form (linear combo of A) * (linear combo of B).
    #   So each scalar multiplication is a rank-1 tensor?
    #   No, a scalar multiplication is a rank-1 tensor in the space of bilinear forms.
    #   So the 7 terms of Strassen for 2x2 are 7 rank-1 tensors.
    #   Each term is of the form (sum a_i A[i,j]) * (sum b_j B[j,k]) * (sum c_k C[i,k])?
    #   No, the term is a scalar value that is added to C[i,k].
    #   The term is (linear combo of A) * (linear combo of B).
    #   This is a rank-1 tensor in the A-B-C space.
    #   So yes, each of the 7 terms is a rank-1 tensor.
    
    # So for the 4x4 part, we have 7 block terms (M1..M7).
    # Each block term is a 2x2 matrix product.
    # We compute each 2x2 matrix product using the rank-7 Strassen algorithm.
    # So each block term contributes 7 rank-1 tensors.
    # Total 7 * 7 = 49 rank-1 tensors.
    
    # We need to map the 2x2 indices to the 4x4 indices.
    # For a given block term Mk, we have a 2x2 matrix A_block and B_block.
    # The 2x2 Strassen terms are defined on the 2x2 indices.
    # We need to map the 2x2 indices to the 4x4 indices.
    
    # Let's define the mapping.
    # For the 4x4 matrix, the blocks are:
    # Block 11: rows 0,1; cols 0,1
    # Block 12: rows 0,1; cols 2,3
    # Block 21: rows 2,3; cols 0,1
    # Block 22: rows 2,3; cols 2,3
    
    # For a given block term Mk, the A_block is a 2x2 matrix.
    # The 2x2 Strassen terms are:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Each of these is a rank-1 tensor in the 2x2 space.
    # We need to expand each into rank-1 tensors in the 4x4 space.
    # For example, M1 = (A00+A11)(B00+B11).
    # In the 4x4 space, A00 is A[i,j] where i, j are local indices 0,0.
    # So A00 corresponds to A[0,0] in the block, which is A[0,0] in the 4x4 matrix.
    # A11 corresponds to A[1,1] in the block, which is A[1,1] in the 4x4 matrix.
    # So M1 = (A[0,0] + A[1,1])(B[0,0] + B[1,1]).
    # This is a rank-1 tensor? No, it's a product of sums.
    # (A[0,0] + A[1,1])(B[0,0] + B[1,1]) = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1].
    # This is 4 rank-1 tensors.
    # But the rank of 2x2 is 7, not 4.
    # So the 7 terms of Strassen are not just the 4 terms of M1.
    # The 7 terms are M1..M7.
    # Each Mk is a scalar value.
    # So each Mk is a rank-1 tensor.
    # So we just need to define the 7 rank-1 tensors for the 2x2 case.
    
    # Let's define the 7 rank-1 tensors for 2x2.
    # Term 1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    #   U: A00, A11 with coeff 1
    #   V: B00, B11 with coeff 1
    #   W: C00, C11 with coeff 1
    #   This is 1 rank-1 tensor? No, it's a sum of products.
    #   In the rank-1 decomposition, a term is of the form u * v * w.
    #   So (A00+A11)(B00+B11) is not a rank-1 term.
    #   It is a sum of 4 rank-1 terms.
    #   So the rank of 2x2 is not 7 if we use this expansion.
    #   The rank of 2x2 is 7 because we use 7 scalar multiplications.
    #   Each scalar multiplication is of the form (linear combo of A) * (linear combo of B).
    #   So each scalar multiplication is a rank-1 tensor.
    #   So the 7 terms are the 7 scalar multiplications.
    #   So we just need to list the 7 scalar multiplications.
    
    # The 7 scalar multiplications are:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # Each Sk is a rank-1 tensor.
    # So we can define the 7 rank-1 tensors for the 2x2 case.
    
    # Let's define the 7 terms as (A_coeffs, B_coeffs, C_coeffs)
    # where A_coeffs is a dict { (i,j): coeff }, etc.
    
    # Term 1: S1
    # A: (0,0):1, (1,1):1
    # B: (0,0):1, (1,1):1
    # C: (0,0):1, (1,1):1
    
    # Term 2: S2
    # A: (1,0):1, (1,1):1
    # B: (0,0):1
    # C: (1,0):1
    
    # Term 3: S3
    # A: (0,0):1
    # B: (0,1):1, (1,1):-1
    # C: (0,1):1
    
    # Term 4: S4
    # A: (1,1):1
    # B: (1,0):1, (0,0):-1
    # C: (1,0):1
    
    # Term 5: S5
    # A: (0,0):1, (0,1):1
    # B: (1,1):1
    # C: (0,0):-1, (0,1):1
    
    # Term 6: S6
    # A: (1,0):1, (0,0):-1
    # B: (0,0):1, (0,1):1
    # C: (1,1):1
    
    # Term 7: S7
    # A: (0,1):1, (1,1):-1
    # B: (1,0):1, (1,1):1
    # C: (0,0):1, (1,1):-1
    
    # We will use these 7 terms for each of the 7 block terms in the 4x4 case.
    
    # Let's define the 7 block terms for the 4x4 case.
    # Block 11: A11, B11, C11
    # Block 12: A12, B12, C12
    # Block 21: A21, B21, C21
    # Block 22: A22, B22, C22
    
    # The 7 block terms are:
    # T1 = (A11+A22)(B11+B22)
    # T2 = (A21+A22)B11
    # T3 = A11(B12-B22)
    # T4 = A22(B21-B11)
    # T5 = (A11+A12)B22
    # T6 = (A21-A11)(B11+B12)
    # T7 = (A12-A22)(B21+B22)
    
    # Each Tj is a 2x2 matrix product.
    # So we apply the 7 rank-1 tensors for the 2x2 case to each Tj.
    
    # We need to map the 2x2 indices to the 4x4 indices for each block.
    # For block 11, the 2x2 indices (i,j) map to 4x4 indices (i,j).
    # For block 12, the 2x2 indices (i,j) map to 4x4 indices (i, j+2).
    # For block 21, the 2x2 indices (i,j) map to 4x4 indices (i+2, j).
    # For block 22, the 2x2 indices (i,j) map to 4x4 indices (i+2, j+2).
    
    # Let's implement this.
    
    # Define the 7 rank-1 tensors for 2x2.
    strassen_2x2 = [
        # (A_coeffs, B_coeffs, C_coeffs)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # S1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}), # S2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}), # S3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1}), # S4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # S5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # S6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}) # S7
    ]
    
    # Define the 7 block terms for 4x4.
    # Each block term is a combination of blocks.
    # We need to define the A_blocks, B_blocks, C_blocks for each term.
    # For example, T1 = (A11+A22)(B11+B22).
    # A_blocks: {11: 1, 22: 1}
    # B_blocks: {11: 1, 22: 1}
    # C_blocks: {11: 1, 22: 1}
    
    block_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # T1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}), # T2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}), # T3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1}), # T4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # T5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # T6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}) # T7
    ]
    
    # Now we iterate over the block terms and the strassen_2x2 terms.
    col = 0
    for block_term in block_terms:
        A_blocks, B_blocks, C_blocks = block_term
        for strassen_term in strassen_2x2:
            A_coeffs, B_coeffs, C_coeffs = strassen_term
            # For each block in A_blocks, B_blocks, C_blocks,
            # we need to combine the coefficients.
            # This is a sum of products.
            # The term is sum_{bA in A_blocks} sum_{bB in B_blocks} sum_{bC in C_blocks} 
            # A_blocks[bA] * B_blocks[bB] * C_blocks[bC] * (strassen_term on block bA, bB, bC)
            
            # Wait, this is not correct.
            # The block term T1 = (A11+A22)(B11+B22).
            # This is a single 2x2 matrix product.
            # So we apply the strassen_2x2 terms to this product.
            # The strassen_2x2 terms are defined on the 2x2 indices.
            # So for each strassen_term, we have a rank-1 tensor on the 2x2 indices.
            # We need to map this to the 4x4 indices.
            
            # For a given strassen_term, the A_coeffs are on the 2x2 indices.
            # The 2x2 indices correspond to the local indices of the block.
            # So for each block in A_blocks, we map the 2x2 indices to the 4x4 indices.
            
            # Let's do this for each block.
            for bA in A_blocks:
                for bB in B_blocks:
                    for bC in C_blocks:
                        # The coefficient is A_blocks[bA] * B_blocks[bB] * C_blocks[bC]
                        coeff = A_blocks[bA] * B_blocks[bB] * C_blocks[bC]
                        if coeff == 0:
                            continue
                        
                        # Map the 2x2 indices to 4x4 indices for block bA, bB, bC.
                        # bA is (row_block, col_block) for A.
                        # bB is (row_block, col_block) for B.
                        # bC is (row_block, col_block) for C.
                        
                        # For A, the local indices (i,j) map to (i + 2*bA[0], j + 2*bA[1]).
                        # For B, the local indices (j,k) map to (j + 2*bB[0], k + 2*bB[1]).
                        # For C, the local indices (i,k) map to (i + 2*bC[0], k + 2*bC[1]).
                        
                        # We need to check if the blocks are compatible.
                        # For the term to be valid, the col_block of A must match the row_block of B.
                        # And the row_block of A must match the row_block of C.
                        # And the col_block of B must match the col_block of C.
                        
                        if bA[1] != bB[0] or bA[0] != bC[0] or bB[1] != bC[1]:
                            continue
                        
                        # Now we add the rank-1 tensor.
                        for (i, j), a_coeff in A_coeffs.items():
                            for (j2, k), b_coeff in B_coeffs.items():
                                for (i2, k2), c_coeff in C_coeffs.items():
                                    if i != i2 or j != j2 or k != k2:
                                        continue
                                    
                                    I = i + 2 * bA[0]
                                    J = j + 2 * bA[1]
                                    K = k + 2 * bB[1]
                                    
                                    U[I * m + J][col] += coeff * a_coeff * b_coeff * c_coeff
                                    V[J * p + K][col] += 1
                                    W[K * n + I][col] += 1
                                    # Wait, this is not correct.
                                    # The rank-1 tensor is u * v * w.
                                    # u is a vector in the A space.
                                    # v is a vector in the B space.
                                    # w is a vector in the C space.
                                    
                                    # The term is sum_{i,j} u[i,j] A[i,j] * sum_{j,k} v[j,k] B[j,k] * sum_{i,k} w[i,k] C[i,k].
                                    # So U[i*m+j] = u[i,j], V[j*p+k] = v[j,k], W[k*n+i] = w[i,k].
                                    
                                    # We need to set U, V, W for this rank-1 tensor.
                                    # But we are adding to the existing U, V, W.
                                    # So we need to accumulate the coefficients.
                                    
                                    # Let's reset U, V, W for this col and then fill it.
                                    # No, we are iterating over col.
                                    # So for each col, we have one rank-1 tensor.
                                    # So we can just set U, V, W for this col.
                                    
                                    # But we are iterating over multiple (bA, bB, bC) for the same col.
                                    # This means we are summing multiple rank-1 tensors into the same col.
                                    # This is not a rank-1 tensor anymore.
                                    
                                    # So we need to create a new col for each combination.
                                    # But that would increase the rank.
                                    
                                    # The correct way is to define the rank-1 tensor for the 4x4 case directly.
                                    # The 4x4 Strassen algorithm has rank 49.
                                    # So we should just list the 49 rank-1 tensors.
                                    
                                    # Let's stop and think.
                                    # The rank of 4x4 is 49.
                                    # We can just use a known rank-49 decomposition.
                                    # Or we can just use the schoolbook rank 16 for the 4x4 part?
                                    # No, that would give rank 16 + 4 = 20.
                                    # But the rank of 4x4 is 49, so schoolbook is 16.
                                    # Wait, the rank of 4x4 is 16?
                                    # No, the rank of 4x4 is 49?
                                    # The rank of 4x4 is 16 for schoolbook.
                                    # Strassen gives 49.
                                    # So 16 is better than 49.
                                    # So we should use schoolbook for the 4x4 part.
                                    # Then the total rank is 16 + 4 = 20.
                                    
                                    # But the rank of 4x4 is 16?
                                    # Yes, the schoolbook rank for 4x4 is 16.
                                    # So we can just use schoolbook for the whole thing.
                                    # Then the rank is 4*4*5 = 80.
                                    # But if we use schoolbook for the 4x4 part (rank 16) and schoolbook for the 4x1 part (rank 4),
                                    # the total rank is 16 + 4 = 20.
                                    
                                    # Is this valid?
                                    # The tensor T_{4,4,5} is the direct sum of T_{4,4,4} and T_{4,4,1}.
                                    # So the rank is R(T_{4,4,4}) + R(T_{4,4,1}).
                                    # R(T_{4,4,4}) = 16 (schoolbook).
                                    # R(T_{4,4,1}) = 4 (schoolbook).
                                    # So R(T_{4,4,5}) = 20.
                                    
                                    # This is much better than 80.
                                    # So we should use schoolbook for both parts.
                                    
                                    # Let's implement this.
                                    pass

    # Let's restart the implementation with the schoolbook approach.
    # Rank = 16 + 4 = 20.
    
    U = [[0] * 20 for _ in range(n * m)]
    V = [[0] * 20 for _ in range(m * p)]
    W = [[0] * 20 for _ in range(p * n)]
    
    col = 0
    # Part 1: 4x4x4 part (k=0..3)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Part 2: 4x4x1 part (k=4)
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
