"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block indices for A (4x4): 2x2 blocks of size 2x2
    # A[0,0] -> i in [0,1], j in [0,1]
    # A[0,1] -> i in [0,1], j in [2,3]
    # A[1,0] -> i in [2,3], j in [0,1]
    # A[1,1] -> i in [2,3], j in [2,3]
    
    # Block indices for B (4x6): 2x3 blocks of size 2x2
    # B[0,c] -> j in [0,1], k in [2c, 2c+1]
    # B[1,c] -> j in [2,3], k in [2c, 2c+1]
    
    # Block indices for C (4x6): 2x3 blocks of size 2x2
    # C[0,c] -> i in [0,1], k in [2c, 2c+1]
    # C[1,c] -> i in [2,3], k in [2c, 2c+1]
    
    # Strassen formulas for 2x2 * 2x1 block multiplication (padded to 2x2):
    # M1 = (A00 + A11) (B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B01 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5  (Note: In 2x2*2x2, C01 is top-right. Here B11 is zero-padded, so C01 is effectively 0? 
    # Wait. The block multiplication is:
    # [C00]   [A00 A01] [B00]
    # [C10] = [A10 A11] [B10]
    # 
    # If we pad B with B11=0 and B01=0? No, B is 2x1.
    # Let's view it as [B00; B10] multiplied by [1 0; 0 0]? No.
    # It's [A00 A01] * [B00; B10] = A00*B00 + A01*B10.
    # This is not a standard square Strassen case directly unless we embed it.
    # However, we can just use the 7 terms of Strassen for the full 2x2*2x2 and ignore the terms that would contribute to the zero-padded parts.
    # Or simpler: Just define the 7 rank-1 tensors that compute the 4 products A00B00, A01B10, A10B00, A11B10 using Strassen's linear combinations.
    # Actually, the 7 Strassen terms compute:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # If we set B01=0 and B11=0 (since we only have B00, B10 in the 2x1 block), then:
    # P1 = (A00+A11)B00
    # P2 = (A10+A11)B00
    # P3 = A00*B10
    # P4 = -A11*B00
    # P5 = 0
    # P6 = (A10-A00)B00
    # P7 = -A11*B10
    
    # This doesn't seem to simplify nicely to just 7 terms for the 2x1 case directly without checking dependencies.
    # Let's stick to the embedding argument:
    # The tensor for 2x2*2x1 matrix multiplication (with 2x2 entries) is a sub-tensor of 2x2*2x2.
    # The rank of 2x2*2x2 is 7.
    # Can we achieve rank 7 for 2x2*2x1?
    # The lower bound for 2x2*2x1 is likely lower, but 7 is a safe upper bound derived from Strassen.
    # Let's verify if we can just use the 7 Strassen terms and set the "missing" B blocks to 0 in the U,V,W construction.
    
    # For each block column c (0, 1, 2):
    for c in range(3):
        # Indices for B and C columns: k_start = 2*c, k_end = 2*c + 2
        k0 = 2 * c
        k1 = 2 * c + 1
        
        # We will use 7 columns in U, V, W for this block c.
        # col_offset = c * 7
        
        # Define the 7 Strassen terms.
        # We need to map A, B, C block entries to specific indices.
        
        # A blocks:
        # A00: i in [0,1], j in [0,1]
        # A01: i in [0,1], j in [2,3]
        # A10: i in [2,3], j in [0,1]
        # A11: i in [2,3], j in [2,3]
        
        # B blocks (for this c):
        # B00: j in [0,1], k in [k0, k1]
        # B10: j in [2,3], k in [k0, k1]
        # B01, B11 are "zero" in the padded sense, so we don't allocate indices for them in V, 
        # but we must ensure the Strassen formulas that use them evaluate to 0 or are handled correctly.
        # Wait, if we use the full Strassen formulas, we need to represent B01 and B11.
        # But B01 and B11 correspond to columns that don't exist in our 4x6 matrix for this specific block column c?
        # No, B is 4x6. The block column c uses columns 2c and 2c+1.
        # The "B01" and "B11" in the Strassen formula refer to the second block column of the 2x2 B matrix.
        # In our case, we are only multiplying by the first block column [B00; B10].
        # So we can set the entries corresponding to B01 and B11 to 0 in the V matrix for these rank-1 terms.
        
        # Let's define the 7 terms explicitly.
        # Each term r (0..6) will have:
        # - Coefficients in U for A entries
        # - Coefficients in V for B entries
        # - Coefficients in W for C entries
        
        # Term 1: P1 = (A00 + A11) * (B00 + B11)
        # Since B11 is 0, P1 = (A00 + A11) * B00
        # U: A00 (+1), A11 (+1)
        # V: B00 (+1)
        # W: C00 (+1), C10 (+1)  -- Wait, P1 contributes to C00 and C10?
        # In 2x2*2x2:
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # In our 2x2*2x1 case (C01, C11 are 0 or not computed):
        # We only care about C00 and C10.
        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        
        # Let's check if the Strassen combination yields this.
        # If we set B01=0, B11=0:
        # P1 = (A00+A11)B00
        # P2 = (A10+A11)B00
        # P3 = A00*B10
        # P4 = -A11*B00
        # P5 = 0
        # P6 = (A10-A00)B00
        # P7 = -A11*B10
        
        # C00 = P1 + P4 - P5 + P7 = (A00+A11)B00 - A11*B00 - 0 - A11*B10 = A00*B00 - A11*B10. 
        # This is WRONG. We want A00*B00 + A01*B10.
        
        # The issue is that the standard Strassen formulas assume a full 2x2 B.
        # We are multiplying [A00 A01] by [B00; B10].
        # This is NOT the same as [A00 A01] by [B00 0; B10 0].
        # If B = [B00 0; B10 0], then:
        # C00 = A00*B00 + A01*B10
        # C01 = A00*0 + A01*0 = 0
        # C10 = A10*B00 + A11*B10
        # C11 = A10*0 + A11*0 = 0
        
        # So yes, we can pad B with zeros in the second column.
        # Then the Strassen formulas apply.
        # Let's re-evaluate the Strassen terms with B01=0, B11=0.
        
        # P1 = (A00+A11)(B00+0) = (A00+A11)B00
        # P2 = (A10+A11)B00
        # P3 = A00(B10-0) = A00*B10
        # P4 = A11(0-B00) = -A11*B00
        # P5 = (A00+A01)*0 = 0
        # P6 = (A10-A00)(B00+0) = (A10-A00)B00
        # P7 = (A01-A11)(B10+0) = (A01-A11)B10
        
        # Now compute C00 and C10:
        # C00 = P1 + P4 - P5 + P7
        #     = (A00+A11)B00 - A11*B00 - 0 + (A01-A11)B10
        #     = A00*B00 + A01*B10 - A11*B10
        # This is STILL wrong. We want A00*B00 + A01*B10.
        # The term -A11*B10 is extra.
        
        # Why? Because Strassen is for square matrices.
        # The formula C00 = P1 + P4 - P5 + P7 is derived for the top-left block of the product.
        # If B is [B00 0; B10 0], the product is correct.
        # Let's check the derivation.
        # C00 = A00B00 + A01B10.
        # My expansion gave A00B00 + A01B10 - A11B10.
        # There is a mismatch.
        
        # Let's re-read Strassen.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # If B01=0, B11=0:
        # M1 = (A00+A11)B00
        # M4 = A11(-B00) = -A11B00
        # M5 = 0
        # M7 = (A01-A11)B10
        
        # C00 = (A00+A11)B00 - A11B00 + (A01-A11)B10
        #     = A00B00 + A01B10 - A11B10
        
        # It seems Strassen does NOT directly work for rectangular matrices by just padding with zeros in this way?
        # Or did I make a mistake in the Strassen formula?
        # Let's check C11.
        # C11 = M1 - M2 + M3 + M6
        #     = (A00+A11)B00 - (A10+A11)B00 + A00B10 + (A10-A00)B00
        #     = A00B00 + A11B00 - A10B00 - A11B00 + A00B10 + A10B00 - A00B00
        #     = A00B10.
        # But C11 should be 0.
        
        # So padding with zeros does NOT make the other blocks zero.
        # Therefore, we cannot simply use the 7 Strassen terms for a 2x2*2x1 product.
        
        # We must find a decomposition for 2x2*2x1 (with 2x2 entries) of rank <= 7?
        # Or just use the schoolbook for the block level?
        # Schoolbook for 2x2*2x1 requires 4 multiplications.
        # Each multiplication is 2x2*2x2, rank 7.
        # Total rank = 4 * 7 = 28 per block column.
        # Total rank = 3 * 28 = 84.
        # This is better than 96, but not 21.
        
        # Can we do better than 28 for 2x2*2x1?
        # The tensor is T_{i,j,k} with i in {0,1} (C row), j in {0,1} (A col/B row), k in {0,1} (C col/B col).
        # Wait, the "entries" are 2x2 matrices.
        # So we are decomposing the tensor of size 2 (C row) x 2 (A col) x 1 (C col) ... no.
        # The indices are:
        # i_C in {0,1} (block row of C)
        # i_A in {0,1} (block row of A)
        # j_A in {0,1} (block col of A)
        # j_B in {0,1} (block row of B)
        # k_C in {0,1} (block col of C)
        # k_B in {0,1} (block col of B)
        
        # Actually, it's simpler.
        # We need to compute C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # where A00, A01, A10, A11, B0, B1, C0, C1 are 2x2 matrices.
        
        # This is equivalent to multiplying a 2x2 matrix of 2x2 blocks by a 2x1 matrix of 2x2 blocks.
        # We can view this as two independent 2x2*2x2 multiplications? No.
        
        # Let's use the known result: The rank of 2x2*2x1 matrix multiplication (scalar) is 3?
        # No, 2x2*2x1 is just 4 scalar multiplications.
        # But here the "scalars" are 2x2 matrices.
        # So it's 4 matrix multiplications of 2x2 matrices.
        # Can we combine them?
        # Yes, if we can find a lower rank decomposition for the block-level tensor.
        # The block-level tensor is of size 2 (C rows) x 2 (A cols) x 1 (C cols) ... no, the indices are mixed.
        
        # Let's just use the schoolbook for the blocks, which gives rank 84.
        # But wait, we can use Strassen for the 2x2 matrix multiplications.
        # Each of the 4 block multiplications costs rank 7.
        # Total 28 per block column.
        # Total 84.
        
        # Is there a way to do better?
        # For 4x4x6, the best known rank is likely around 40-50?
        # But 84 is a safe improvement over 96.
        
        # Let's implement the 84 rank solution.
        
    # Reset U, V, W for rank 84
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for c in range(3):
        k0 = 2 * c
        k1 = 2 * c + 1
        
        # 4 block multiplications:
        # C00 += A00 B0
        # C00 += A01 B1
        # C10 += A10 B0
        # C10 += A11 B1
        
        # Each is a 2x2*2x2 matrix multiplication, using Strassen (rank 7).
        
        # Helper to add a Strassen decomposition for A_block (2x2 indices in U) 
        # times B_block (2x2 indices in V) 
        # to C_block (2x2 indices in W)
        
        def add_strassen(U, V, W, rank, col, 
                         a_i0, a_i1, a_j0, a_j1, # A block row/col indices in U
                         b_j0, b_j1, b_k0, b_k1, # B block row/col indices in V
                         c_k0, c_k1, c_i0, c_i1): # C block col/row indices in W
            
            # A00: i in [a_i0, a_i0+1], j in [a_j0, a_j0+1]
            # A01: i in [a_i0, a_i0+1], j in [a_j0+2, a_j0+3]
            # A10: i in [a_i0+2, a_i0+3], j in [a_j0, a_j0+1]
            # A11: i in [a_i0+2, a_i0+3], j in [a_j0+2, a_j0+3]
            
            # B00: j in [b_j0, b_j0+1], k in [b_k0, b_k0+1]
            # B01: j in [b_j0, b_j0+1], k in [b_k0+2, b_k0+3]
            # B10: j in [b_j0+2, b_j0+3], k in [b_k0, b_k0+1]
            # B11: j in [b_j0+2, b_j0+3], k in [b_k0+2, b_k0+3]
            
            # C00: k in [c_k0, c_k0+1], i in [c_i0, c_i0+1]
            # C01: k in [c_k0, c_k0+1], i in [c_i0+2, c_i0+3]
            # C10: k in [c_k0+2, c_k0+3], i in [c_i0, c_i0+1]
            # C11: k in [c_k0+2, c_k0+3], i in [c_i0+2, c_i0+3]
            
            # Strassen terms
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            for r in range(7):
                idx = col + r
                
                # Determine coefficients for U, V, W for this term
                
                # Term 1
                if r == 0:
                    # U: A00 (+1), A11 (+1)
                    for i in range(2):
                        for j in range(2):
                            U[a_i0 + i][a_j0 + j] += 1
                            U[a_i0 + 2 + i][a_j0 + 2 + j] += 1
                    # V: B00 (+1), B11 (+1)
                    for j in range(2):
                        for k in range(2):
                            V[b_j0 + j][b_k0 + k] += 1
                            V[b_j0 + 2 + j][b_k0 + 2 + k] += 1
                    # W: C00 (+1), C11 (+1)
                    for k in range(2):
                        for i in range(2):
                            W[c_k0 + k][c_i0 + i] += 1
                            W[c_k0 + 2 + k][c_i0 + 2 + i] += 1
                            
                # Term 2
                elif r == 1:
                    # U: A10 (+1), A11 (+1)
                    for i in range(2):
                        for j in range(2):
                            U[a_i0 + 2 + i][a_j0 + j] += 1
                            U[a_i0 + 2 + i][a_j0 + 2 + j] += 1
                    # V: B00 (+1)
                    for j in range(2):
                        for k in range(2):
                            V[b_j0 + j][b_k0 + k] += 1
                    # W: C10 (+1), C11 (-1)
                    for k in range(2):
                        for i in range(2):
                            W[c_k0 + 2 + k][c_i0 + i] += 1
                            W[c_k0 + 2 + k][c_i0 + 2 + i] -= 1

                # Term 3
                elif r == 2:
                    # U: A00 (+1)
                    for i in range(2):
                        for j in range(2):
                            U[a_i0 + i][a_j0 + j] += 1
                    # V: B10 (+1), B11 (-1)
                    for j in range(2):
                        for k in range(2):
                            V[b_j0 + 2 + j][b_k0 + k] += 1
                            V[b_j0 + 2 + j][b_k0 + 2 + k] -= 1
                    # W: C01 (+1), C11 (+1)
                    for k in range(2):
                        for i in range(2):
                            W[c_k0 + k][c_i0 + 2 + i] += 1
                            W[c_k0 + 2 + k][c_i0 + 2 + i] += 1

                # Term 4
                elif r == 3:
                    # U: A11 (+1)
                    for i in range(2):
                        for j in range(2):
                            U[a_i0 + 2 + i][a_j0 + 2 + j] += 1
                    # V: B01 (+1), B00 (-1)
                    for j in range(2):
                        for k in range(2):
                            V[b_j0 + j][b_k0 + 2 + k] += 1
                            V[b_j0 + j][b_k0 + k] -= 1
                    # W: C00 (+1), C10 (+1)
                    for k in range(2):
                        for i in range(2):
                            W[c_k0 + k][c_i0 + i] += 1
                            W[c_k0 + 2 + k][c_i0 + i] += 1

                # Term 5
                elif r == 4:
                    # U: A00 (+1), A01 (+1)
                    for i in range(2):
                        for j in range(2):
                            U[a_i0 + i][a_j0 + j] += 1
                            U[a_i0 + i][a_j0 + 2 + j] += 1
                    # V: B11 (+1)
                    for j in range(2):
                        for k in range(2):
                            V[b_j0 + 2 + j][b_k0 + 2 + k] += 1
                    # W: C00 (-1), C01 (+1)
                    for k in range(2):
                        for i in range(2):
                            W[c_k0 + k][c_i0 + i] -= 1
                            W[c_k0 + k][c_i0 + 2 + i] += 1

                # Term 6
                elif r == 5:
                    # U: A10 (+1), A00 (-1)
                    for i in range(2):
                        for j in range(2):
                            U[a_i0 + 2 + i][a_j0 + j] += 1
                            U[a_i0 + i][a_j0 + j] -= 1
                    # V: B00 (+1), B01 (+1)
                    for j in range(2):
                        for k in range(2):
                            V[b_j0 + j][b_k0 + k] += 1
                            V[b_j0 + j][b_k0 + 2 + k] += 1
                    # W: C11 (+1)
                    for k in range(2):
                        for i in range(2):
                            W[c_k0 + 2 + k][c_i0 + 2 + i] += 1

                # Term 7
                elif r == 6:
                    # U: A01 (+1), A11 (-1)
                    for i in range(2):
                        for j in range(2):
                            U[a_i0 + i][a_j0 + 2 + j] += 1
                            U[a_i0 + 2 + i][a_j0 + 2 + j] -= 1
                    # V: B10 (+1), B11 (+1)
                    for j in range(2):
                        for k in range(2):
                            V[b_j0 + 2 + j][b_k0 + k] += 1
                            V[b_j0 + 2 + j][b_k0 + 2 + k] += 1
                    # W: C00 (+1)
                    for k in range(2):
                        for i in range(2):
                            W[c_k0 + k][c_i0 + i] += 1

            return col + 7

        # Block 0: C00 += A00 B0
        # A00: i 0-1, j 0-1
        # B0: j 0-1, k k0-k1
        # C00: i 0-1, k k0-k1
        col = add_strassen(U, V, W, rank, col, 
                           0, 0, 0, 0, 
                           0, 0, k0, 0, 
                           k0, 0, 0, 0)
        
        # Block 1: C00 += A01 B1
        # A01: i 0-1, j 2-3
        # B1: j 2-3, k k0-k1
        # C00: i 0-1, k k0-k1
        col = add_strassen(U, V, W, rank, col, 
                           0, 0, 2, 0, 
                           2, 0, k0, 0, 
                           k0, 0, 0, 0)
        
        # Block 2: C10 += A10 B0
        # A10: i 2-3, j 0-1
        # B0: j 0-1, k k0-k1
        # C10: i 2-3, k k0-k1
        col = add_strassen(U, V, W, rank, col, 
                           2, 0, 0, 0, 
                           0, 0, k0, 0, 
                           k0, 0, 2, 0)
        
        # Block 3: C10 += A11 B1
        # A11: i 2-3, j 2-3
        # B1: j 2-3, k k0-k1
        # C10: i 2-3, k k0-k1
        col = add_strassen(U, V, W, rank, col, 
                           2, 0, 2, 0, 
                           2, 0, k0, 0, 
                           k0, 0, 2, 0)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
