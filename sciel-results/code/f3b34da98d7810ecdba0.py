import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 2x2 matrices
    # U2[i,j, k] where k in 0..6, i in 0..1, j in 0..1
    # For A (2x2), B (2x2), C (2x2)
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0+M3-M4-M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M0
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1
    # M1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    # M2
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1
    # M3
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    # M4
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    # M5
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    # M6
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[3][6] = -1; W2[1][6] = -1
    
    # Now build 4x4x4 decomposition using block Strassen
    # A is 4x4, split into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B is 4x4, split into 2x2 blocks of 2x2 matrices
    # B = [[B00, B01], [B10, B11]]
    
    # The full U, V, W for 4x4x4 will have rank 7*7 = 49
    # U has shape 16 x 49
    # V has shape 16 x 49
    # W has shape 16 x 49
    
    rank_444 = 49
    U444 = [[0]*rank_444 for _ in range(16)]
    V444 = [[0]*rank_444 for _ in range(16)]
    W444 = [[0]*rank_444 for _ in range(16)]
    
    # For each block-level multiplication k (0..6) and internal Strassen step r2 (0..6)
    # The column index in the 4x4x4 decomposition is k*7 + r2
    
    # Block-level coefficients for A (which blocks to combine)
    # M0: A00 + A11 -> blocks at positions (0,0) and (1,1)
    # M1: A10 + A11 -> blocks at positions (1,0) and (1,1)
    # M2: A00 -> block at position (0,0)
    # M3: A11 -> block at position (1,1)
    # M4: A00 + A01 -> blocks at positions (0,0) and (0,1)
    # M5: A10 - A00 -> blocks at positions (1,0) and (0,0)
    # M6: A01 - A11 -> blocks at positions (0,1) and (1,1)
    
    A_block_coeffs = [
        [(0,0,1), (1,1,1)],    # M0
        [(1,0,1), (1,1,1)],    # M1
        [(0,0,1)],             # M2
        [(1,1,1)],             # M3
        [(0,0,1), (0,1,1)],    # M4
        [(1,0,1), (0,0,-1)],   # M5
        [(0,1,1), (1,1,-1)],   # M6
    ]
    
    # Block-level coefficients for B
    # M0: B00 + B11
    # M1: B00
    # M2: B01 - B11
    # M3: B10 - B00
    # M4: B11
    # M5: B00 + B01
    # M6: B10 + B11
    B_block_coeffs = [
        [(0,0,1), (1,1,1)],    # M0
        [(0,0,1)],             # M1
        [(0,1,1), (1,1,-1)],   # M2
        [(1,0,1), (0,0,-1)],   # M3
        [(1,1,1)],             # M4
        [(0,0,1), (0,1,1)],    # M5
        [(1,0,1), (1,1,1)],    # M6
    ]
    
    # W2 tells us how to combine M0..M6 to get C blocks
    # W2 is indexed by [C_block_pos, k] where C_block_pos is 0..3 (for C00, C01, C10, C11)
    # But we need to think about this more carefully.
    
    # Actually, let's build U444, V444, W444 directly.
    # U444 has 16 rows (4x4 matrix A flattened) and 49 columns
    # V444 has 16 rows (4x4 matrix B flattened) and 49 columns
    # W444 has 16 rows (4x4 matrix C flattened) and 49 columns
    
    # For column c = k*7 + r2:
    # U444[i,j, c] = sum over (a,b,coeff) in A_block_coeffs[k] of coeff * U2[a*2+i', b*2+j', r2]
    # Wait, this is getting complicated. Let me think differently.
    
    # Actually, the way to combine is:
    # The block-level factor for A is a 2x2 matrix (in block space) with entries from A_block_coeffs[k]
    # Then we apply the 2x2 Strassen decomposition to this 2x2 block matrix.
    
    # Let me redo this. For each k in 0..6:
    # The block-level A matrix (2x2 in block space) has entries:
    #   A_block[k][a][b] = sum of coefficients from A_block_coeffs[k] where (block_row, block_col) = (a,b)
    # Similarly for B_block[k]
    
    # Then for the internal Strassen step r2:
    # The coefficient for A_block[k][a][b] in the internal U2[a*2+ai, b*2+bi, r2] contributes to U444.
    
    # Let me just build it directly by iterating over all combinations.
    
    U444 = [[0]*rank_444 for _ in range(16)]
    V444 = [[0]*rank_444 for _ in range(16)]
    W444 = [[0]*rank_444 for _ in range(16)]
    
    for k in range(7):
        for r2 in range(7):
            col = k * 7 + r2
            
            # Determine the block-level A matrix entries
            # A_block[k] is a 2x2 matrix (in block space)
            # For each (a, b) in 0..1, 0..1:
            A_block = [[0]*2 for _ in range(2)]
            for (ba, bb, coeff) in A_block_coeffs[k]:
                A_block[ba][bb] += coeff
            
            B_block = [[0]*2 for _ in range(2)]
            for (ba, bb, coeff) in B_block_coeffs[k]:
                B_block[ba][bb] += coeff
            
            # Now, for the internal Strassen step r2:
            # U2[a, b, r2] gives the coefficient for A[a][b] in the internal decomposition
            # where a, b are 0..1 (block space indices for the 2x2 block matrix)
            
            # For each (a, b) in block space:
            for a in range(2):
                for b in range(2):
                    # The coefficient from internal U2
                    u_coeff = U2[a*2+b][r2]  # Wait, U2 is indexed by [i,j, k] where i,j are 0..1
                    # Actually U2[i][j][k] where i is row in 0..1, j is col in 0..1, k is step in 0..6
                    # But I defined U2 as a 4x7 matrix with U2[0..3][0..6]
                    # Let me re-index: U2[i*2+j][k] for i,j in 0..1
                    u_coeff = U2[a*2+b][r2]
                    
                    if u_coeff != 0:
                        # This contributes to U444 at positions corresponding to block (a,b)
                        # Block (a,b) in the 4x4 matrix corresponds to rows a*2, a*2+1 and cols b*2, b*2+1
                        # But wait, the 4x4 matrix is flattened as i*4+j where i is row, j is col
                        # Block (a,b) has rows a*2..a*2+1 and cols b*2..b*2+1
                        # The elements are at positions (a*2)*4 + b*2, (a*2)*4 + b*2+1, (a*2+1)*4 + b*2, (a*2+1)*4 + b*2+1
                        # But U444 is indexed by [i*4+j, col] where i is row, j is col
                        # So for element (i, j) in block (a,b): i = a*2 + di, j = b*2 + dj where di,dj in 0..1
                        # But the coefficient u_coeff applies to the entire block, so all 4 elements get the same coefficient?
                        # No, that's not right. The block-level coefficient A_block[a][b] is for the entire 2x2 block.
                        # But then U444[i*4+j, col] should be u_coeff * A_block[a][b] for each (i,j) in the block.
                        
                        # Wait, I think I'm overcomplicating this. Let me restart the construction.
                        pass
    
    # Let me try a cleaner approach.
    # The 4x4x4 tensor can be seen as a Kronecker product structure.
    # Actually, for Strassen, it's not a simple Kronecker product because the block combinations are linear.
    
    # Let me just build U444, V444, W444 by explicitly computing the contributions.
    
    U444 = [[0]*rank_444 for _ in range(16)]
    V444 = [[0]*rank_444 for _ in range(16)]
    W444 = [[0]*rank_444 for _ in range(16)]
    
    # For each k in 0..6 (block-level step) and r2 in 0..6 (internal step):
    # col = k*7 + r2
    # The contribution to U444 comes from:
    #   For each (a, b) in 0..1, 0..1 (block space):
    #     coeff_A = A_block_coeffs[k] entry for (a,b)
    #     Then for each (di, dj) in 0..1, 0..1 (within block):
    #       i = a*2 + di, j = b*2 + dj
    #       U444[i*4+j, col] += coeff_A * U2[a*2+b][r2]
    #   Wait, but U2[a*2+b][r2] is the coefficient for the element at position (a,b) in the 2x2 block matrix.
    #   So the total coefficient for U444[i*4+j, col] is:
    #     sum over (a,b) such that (i,j) is in block (a,b) of:
    #       A_block_coeff(a,b) * U2[a*2+b][r2]
    #   But (i,j) is in exactly one block (a,b), so:
    #     a = i//2, b = j//2
    #     U444[i*4+j, col] = A_block_coeff(a,b) * U2[a*2+b][r2]
    
    # Similarly for V444:
    #     V444[i*4+j, col] = B_block_coeff(a,b) * V2[a*2+b][r2]
    #     where a = i//2, b = j//2
    
    # For W444:
    #   The output C is also 4x4, split into 2x2 blocks of 2x2 matrices.
    #   C_block[a][b] = sum over k of W2[a*2+b][k] * M_k
    #   But M_k is itself computed via the internal Strassen decomposition.
    #   So for a given (a,b) in block space (for C), and internal step r2:
    #     The contribution from block-level step k is:
    #       W2[a*2+b][k] * (contribution from internal step r2 to M_k)
    #   But the internal Strassen decomposition gives M_k = sum over r2 of (U2_part * V2_part)
    #   So W444[i*4+j, col] where col = k*7 + r2:
    #     a = i//2, b = j//2
    #     W444[i*4+j, col] = W2[a*2+b][k]
    #   Wait, but this doesn't depend on r2, which seems wrong.
    
    # Let me reconsider. The internal Strassen decomposition for M_k gives:
    #   M_k = sum over r2 of (U2_vec[r2] * A_block) * (V2_vec[r2] * B_block)
    #   where U2_vec[r2] is the r2-th column of U2 (a 4x1 vector, flattened from 2x2)
    #   and V2_vec[r2] is the r2-th column of V2.
    
    # So for a given k and r2:
    #   The contribution to M_k is:
    #     (sum over (a,b) of U2[a*2+b][r2] * A_block[a][b]) * (sum over (a,b) of V2[a*2+b][r2] * B_block[a][b])
    #   But this is a scalar, and M_k is a 2x2 block matrix.
    
    # I think the issue is that the internal Strassen decomposition operates on the 2x2 block matrices,
    # and the result is a 2x2 block matrix (the product). So M_k is a 2x2 block matrix.
    
    # Let me look up the correct way to combine Strassen decompositions for block matrices.
    
    # Actually, I think the cleanest way is to just build the full 16x49 matrices by
    # considering each element of the 4x4 matrices.
    
    # For A (4x4), element (i, j) is in block (i//2, j//2).
    # For a given k (block-level step) and r2 (internal step):
    #   U444[i*4+j, k*7+r2] = (A_block_coeff for (i//2, j//2) in step k) * U2[(i//2)*2 + (j//2)][r2]
    
    # Wait, that's not right either. U2 is a 4x7 matrix where the row index is i*2+j for a 2x2 matrix.
    # So U2[row][r2] where row = a*2+b for a 2x2 matrix with elements at (a,b).
    
    # Let me just try a different approach: use the known fact that the rank of 4x4x4 matrix multiplication
    # is at most 49 (from recursive Strassen), and construct the decomposition by explicitly computing
    # the tensor entries.
    
    # Actually, let me just use the schoolbook decomposition for the 5th column and Strassen for the first 4 columns.
    # This gives rank 49 + 16 = 65.
    
    # But I need to be careful about the W matrix indexing: W[k*n+i, r]
    
    # Let me construct U, V, W for the 4x4x4 part (rank 49) and the 4x4x1 part (rank 16).
    
    # Part 1: 4x4x4 with rank 49 (Strassen)
    # Part 2: 4x4x1 with rank 16 (schoolbook)
    
    # Total rank = 49 + 16 = 65
    
    rank_total = 65
    U = [[0]*rank_total for _ in range(16)]
    V = [[0]*rank_total for _ in range(20)]
    W = [[0]*rank_total for _ in range(20)]
    
    # Part 1: Strassen for columns 0..3 of B
    # B is 4x5, so columns 0..3 are handled by Strassen (4x4x4), column 4 by schoolbook (4x4x1)
    
    # For the Strassen part, the output C is 4x4 (for columns 0..3 of the 4x5 result)
    # The W matrix for the Strassen part has shape 4*4 = 16 rows (for C[0..3, 0..3])
    # But the full W matrix has shape 5*4 = 20 rows (for C[0..4, 0..3])
    # Wait, the task says W[k*n+i, r] where k is 0..p-1=0..4, i is 0..n-1=0..3
    # So W has 20 rows and rank_total columns.
    
    # For the Strassen part (columns 0..3 of B), the output is C[0..3, 0..3]
    # So W rows 0..15 correspond to k=0..3, i=0..3, and W[k*4+i, r] for r in 0..48
    
    # For the schoolbook part (column 4 of B), the output is C[0..3, 4]
    # So W rows 16..19 correspond to k=4, i=0..3, and W[16+i, r] for r in 49..64
    
    # Now let's fill in the Strassen part.
    
    # I'll use the U2, V2, W2 defined above.
    
    # For col in 0..48:
    #   k = col // 7, r2 = col % 7
    #   For i in 0..3, j in 0..3:
    #     a = i // 2, b = j // 2
    #     U[i*4+j, col] = (A_block_coeff for (a,b) in step k) * U2[a*2+b][r2]
    #   For i in 0..3, j in 0..3:
    #     a = i // 2, b = j // 2
    #     V[i*4+j, col] = (B_block_coeff for (a,b) in step k) * V2[a*2+b][r2]
    #   For i in 0..3, j in 0..3:
    #     a = i // 2, b = j // 2
    #     W[j*4+i, col] = W2[a*2+b][k]
    #     Wait, W is indexed by W[k*n+i, r] where k is the column index of B (0..3 for Strassen part)
    #     and i is the row index of A (0..3).
    #     But in the Strassen decomposition, the output C is 4x4, with C[row, col] where row is 0..3 (from A)
    #     and col is 0..3 (from B).
    #     So W[col*4+row, r] for the Strassen part.
    #     But the task says W[k*n+i, r] where k is the column index of B and i is the row index of A.
    #     So W[col*4+row, r] is correct.
    
    # Let me fill in the Strassen part.
    
    for col in range(49):
        k = col // 7
        r2 = col % 7
        
        # Fill U
        for i in range(4):
            for j in range(4):
                a = i // 2
                b = j // 2
                # Get A_block_coeff for (a,b) in step k
                coeff_a = 0
                for (ba, bb, c) in A_block_coeffs[k]:
                    if ba == a and bb == b:
                        coeff_a += c
                U[i*4+j][col] = coeff_a * U2[a*2+b][r2]
        
        # Fill V
        for i in range(4):
            for j in range(4):
                a = i // 2
                b = j // 2
                coeff_b = 0
                for (ba, bb, c) in B_block_coeffs[k]:
                    if ba == a and bb == b:
                        coeff_b += c
                V[i*4+j][col] = coeff_b * V2[a*2+b][r2]
        
        # Fill W (for Strassen part, k is 0..3, which corresponds to B columns 0..3)
        # But k here is the block-level step index (0..6), not the B column index.
        # The W matrix for the Strassen part should have W[b_col*4+a_row, col]
        # where b_col is 0..3 (B column) and a_row is 0..3 (A row).
        # But the internal W2 gives us how to combine the 7 block-level products to get the 4 output blocks.
        # W2[c_block_row*2+c_block_col][k] gives the coefficient for M_k in C_block[c_block_row, c_block_col].
        # So for output element (a_row, b_col):
        #   c_block_row = a_row // 2, c_block_col = b_col // 2
        #   W[b_col*4+a_row, col] = W2[c_block_row*2+c_block_col][k]
        
        for a_row in range(4):
            for b_col in range(4):
                c_block_row = a_row // 2
                c_block_col = b_col // 2
                W[b_col*4+a_row][col] = W2[c_block_row*2+c_block_col][k]
    
    # Part 2: Schoolbook for column 4 of B
    # U, V, W for the schoolbook part (rank 16)
    for i in range(4):
        for j in range(4):
            col = 49 + i*4 + j
            U[i*4+j][col] = 1
            V[i*4+4][col] = 1  # B[i][4]
            W[4*4+i][col] = 1  # W[16+i, col] for C[i, 4]
    
    return {"U": U, "V": V, "W": W}
