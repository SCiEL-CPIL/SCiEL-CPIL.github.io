import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # The tensor for multiplying a 4x4 matrix by a 4x8 matrix
    # Can be viewed as two independent 4x4x4 multiplications
    # (Left half of B and Right half of B)
    
    # We use Strassen's algorithm for the 4x4x4 sub-problems.
    # Rank of 4x4x4 via Strassen is 7 * 7 = 49.
    # Total rank for 4x4x8 is 49 + 49 = 98.
    
    total_rank = 98
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * total_rank for _ in range(p * n)]  # 32 x 98
    
    # Helper to add Strassen decomposition for a 4x4x4 multiplication
    # offset: column offset in the final rank dimension
    # b_col_offset: column offset for B in the V matrix (0 for left, 4 for right)
    
    def add_strassen_4x4x4(offset, b_col_offset):
        # Strassen products for 4x4 x 4x4
        # A is 4x4, B is 4x4, C is 4x4
        # Blocks are 2x2
        
        # 7 Products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = (A00-A10)(B10+B11)
        # M4 = A00B00
        # M5 = A11B11
        # M6 = (A01-A11)(B00+B01)
        # M7 = (A00+A01)(B10-B00)
        
        # For each product, we need to decompose the 2x2x2 matrix multiplication.
        # We use Strassen again for the 2x2x2 level? 
        # No, the "scalars" in the first level Strassen are 2x2 matrices.
        # So we need the rank of 2x2x2 tensor multiplication.
        # Rank of 2x2x2 is 7.
        # So each of the 7 products in the 4x4x4 level requires 7 "scalar" multiplications (which are 2x2x2).
        # Wait, this logic is recursive.
        # Level 1: 4x4x4 -> 7 products of 2x2 matrices.
        # Level 2: Each 2x2 matrix multiplication has rank 7.
        # So total rank is 7 * 7 = 49.
        
        # We need to map the indices correctly.
        # Let's define the 7 Strassen steps for 4x4x4.
        # For each step s (0..6), we have linear combinations of A blocks and B blocks.
        # Let A_comb[s] be a 2x2 matrix of coefficients for A blocks (A00, A01, A10, A11).
        # Let B_comb[s] be a 2x2 matrix of coefficients for B blocks (B00, B01, B10, B11).
        # The product is P_s = A_comb[s] * B_comb[s] (2x2 matrix multiplication).
        # We decompose P_s using Strassen 2x2x2.
        
        # Strassen 4x4x4 Coefficients for A (2x2 block indices: row 0..1, col 0..1)
        # A blocks: A00, A01; A10, A11
        # Strassen formulas for 4x4:
        # M1: A00+A11 -> [[1,0],[0,1]]
        # M2: A10+A11 -> [[0,0],[1,1]]
        # M3: A00-A10 -> [[1,0],[-1,0]]
        # M4: A00 -> [[1,0],[0,0]]
        # M5: A11 -> [[0,0],[0,1]]
        # M6: A01-A11 -> [[0,1],[0,-1]]
        # M7: A00+A01 -> [[1,1],[0,0]]
        
        A_coefs = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [-1, 0]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[1, 1], [0, 0]]
        ]
        
        # Strassen formulas for 4x4 B:
        # M1: B00+B11 -> [[1,0],[0,1]]
        # M2: B00 -> [[1,0],[0,0]]
        # M3: B10+B11 -> [[0,0],[1,1]]
        # M4: B00 -> [[1,0],[0,0]]
        # M5: B11 -> [[0,0],[0,1]]
        # M6: B00+B01 -> [[1,1],[0,0]]
        # M7: B10-B00 -> [[0,-1],[1,0]]
        
        B_coefs = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, -1], [1, 0]]
        ]
        
        # Now, for each of these 7 products, we have two 2x2 matrices (linear combos of blocks).
        # We need to multiply them.
        # The multiplication of two 2x2 matrices has rank 7.
        # We use Strassen 2x2x2 for this.
        
        # Strassen 2x2x2 Coefficients for "Scalar" A (which is actually a 2x2 block from the higher level? No.)
        # Let's be careful with indices.
        
        # The "matrix" being multiplied at the 4x4 level is composed of 2x2 blocks.
        # Let's say L_A is a 2x2 matrix where each entry is a 2x2 matrix (a block of the original 4x4 A).
        # Wait, no.
        # In the 4x4 Strassen, L_A is a 2x2 matrix. The entries of L_A are 2x2 matrices (blocks of A).
        # So L_A is effectively a 4x4 matrix, but structured as 2x2 blocks.
        # We are multiplying L_A (4x4) by L_B (4x4)?
        # No.
        # M1 = (A00+A11) * (B00+B11).
        # Here (A00+A11) is a 2x2 matrix. (B00+B11) is a 2x2 matrix.
        # Their product is a 2x2 matrix.
        # This 2x2 matrix product is added to the result C.
        # Specifically, the result of M1 is a 2x2 matrix.
        # This 2x2 matrix corresponds to the top-left block of C? No.
        # In 4x4 Strassen, the 7 products M1..M7 are 2x2 matrices.
        # The final C (4x4) is composed of 2x2 blocks C00, C01, C10, C11.
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        
        # So, for each product s (M_s), we compute the product of two 2x2 matrices.
        # Let A_s be the 2x2 matrix (linear combo of A blocks).
        # Let B_s be the 2x2 matrix (linear combo of B blocks).
        # P_s = A_s * B_s (2x2 matrix).
        # We decompose the multiplication of A_s and B_s.
        # A_s is 2x2, B_s is 2x2.
        # We use Strassen 2x2x2 to compute P_s.
        # This requires 7 "scalar" multiplications.
        # But the "scalars" here are the entries of the 2x2 matrices A_s and B_s.
        # The entries of A_s are linear combinations of entries of the 2x2 blocks of A.
        # i.e., A_s[i,j] is a sum of A_block[I,J][i,j] with coefficients.
        # Actually, A_s IS the 2x2 matrix formed by the blocks.
        # Let's clarify indices.
        
        # Original A is 4x4.
        # A_block[I,J] is the 2x2 submatrix A[I*2:I*2+2, J*2:J*2+2].
        # A_s is a 2x2 matrix.
        # A_s[I,J] = sum_{I',J'} A_coefs[s][I,J] * A_block[I',J'].
        # Wait, A_coefs[s] is 2x2.
        # So A_s[I,J] is a linear combination of the 4 blocks A_block[0,0], A_block[0,1], A_block[1,0], A_block[1,1].
        # Specifically, A_s[I,J] = A_coefs[s][I,J] * A_block[I,J]?
        # No, that would mean A_s has the same structure as the block layout.
        # For M1: A_coefs is [[1,0],[0,1]].
        # So A_s[0,0] = A_block[0,0], A_s[1,1] = A_block[1,1]. A_s[0,1]=0, A_s[1,0]=0.
        # So A_s is diag(A00, A11).
        # B_s is diag(B00, B11).
        # P_s = A_s * B_s.
        # P_s[0,0] = A00 * B00.
        # P_s[1,1] = A11 * B11.
        # P_s[0,1] = 0.
        # P_s[1,0] = 0.
        # This P_s is a 2x2 matrix of 2x2 matrices?
        # No. A00 is a 2x2 matrix. B00 is a 2x2 matrix.
        # A00 * B00 is a 2x2 matrix.
        # So P_s[0,0] is a 2x2 matrix.
        # P_s is a 2x2 matrix where each entry is a 2x2 matrix.
        # This is effectively a 4x4 matrix.
        # This P_s is added to the corresponding block of C.
        
        # So, we need to decompose the multiplication of two 2x2 matrices (A_s and B_s).
        # But A_s and B_s are NOT scalar. They are 2x2 matrices.
        # So we are performing a 4x4 matrix multiplication?
        # No.
        # A_s is a 2x2 matrix. The "entries" of A_s are 2x2 matrices.
        # This is exactly the structure of a 4x4 matrix multiplication if we view the 2x2 blocks as scalars.
        # So we are back to multiplying 4x4 matrices?
        # No, the "scalars" are 2x2 matrices.
        # This is recursive.
        # To multiply two 2x2 matrices where entries are 2x2 matrices, we use Strassen on the 2x2 structure.
        # This gives 7 products of "2x2 matrices".
        # But the "2x2 matrices" here are the blocks.
        # So we need to multiply two 2x2 matrices (the blocks).
        # This is a 2x2x2 tensor multiplication.
        # Rank of 2x2x2 is 7.
        # So total rank for 4x4x4 is 7 (outer Strassen) * 7 (inner Strassen) = 49.
        
        # Let's implement this recursively.
        # Function to add a 2x2x2 Strassen decomposition.
        # Inputs:
        # - U, V, W matrices (global)
        # - col_offset: offset in rank dimension
        # - a_row_offset, a_col_offset: offsets for A in U
        #   (Actually, A is fixed 4x4 for this subproblem, so we can just use local 4x4 indices mapped to global)
        #   Let's define a helper that maps local 4x4 indices to global indices.
        
        # Let's define the 2x2x2 Strassen coefficients.
        # For 2x2 matrices X, Y, Z = X*Y.
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = (X00-X10)(Y10+Y11)
        # M4 = X00Y00
        # M5 = X11Y11
        # M6 = (X01-X11)(Y00+Y01)
        # M7 = (X00+X01)(Y10-Y00)
        
        # Z00 = M1 + M4 - M5 + M6
        # Z01 = M4 + M5
        # Z10 = M2 + M3
        # Z11 = M1 - M2 + M3 + M5
        
        # Coefficients for X (2x2):
        X_coefs = [
            [[1, 0], [0, 1]], # M1
            [[0, 0], [1, 1]], # M2
            [[1, 0], [-1, 0]],# M3
            [[1, 0], [0, 0]], # M4
            [[0, 0], [0, 1]], # M5
            [[0, 1], [0, -1]],# M6
            [[1, 1], [0, 0]]  # M7
        ]
        
        # Coefficients for Y (2x2):
        Y_coefs = [
            [[1, 0], [0, 1]], # M1
            [[1, 0], [0, 0]], # M2
            [[0, 0], [1, 1]], # M3
            [[1, 0], [0, 0]], # M4
            [[0, 0], [0, 1]], # M5
            [[1, 1], [0, 0]], # M6
            [[0, -1], [1, 0]] # M7
        ]
        
        # Reconstruction for Z (2x2):
        # Z[r,c] = sum_{s} recon[s][r,c] * M_s
        recon = [
            [[1, 0], [0, 0]], # M1 -> Z00, Z11
            [[0, 0], [0, 0]], # M2 -> Z10, Z11
            [[0, 0], [0, 0]], # M3 -> Z10, Z11
            [[1, 1], [0, 0]], # M4 -> Z00, Z01
            [[-1, 1], [0, 1]],# M5 -> Z00, Z01, Z11
            [[1, 0], [0, 0]], # M6 -> Z00
            [[0, 0], [0, 0]]  # M7 -> not used in standard Strassen reconstruction?
        ]
        # Wait, M7 is not in the reconstruction above?
        # Standard Strassen:
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        # M7 is indeed not used?
        # Let's check.
        # There are 7 products.
        # If M7 is not used, we only have 6 products contributing?
        # No, that would imply rank 6, which is impossible for 2x2x2.
        # I must be missing a term or using a non-standard variant.
        # Let's use a known valid Strassen decomposition.
        
        # One standard set:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = (A11-A21)(B21+B22)
        # M4 = A11B11
        # M5 = A22B22
        # M6 = (A12-A22)(B11+B12)
        # M7 = (A11+A12)(B21-B11)
        
        # C11 = M1 + M4 - M5 + M6
        # C12 = M4 + M5
        # C21 = M2 + M3
        # C22 = M1 - M2 + M3 + M5
        
        # Again, M7 is missing from reconstruction.
        # This suggests my list of Strassen products is wrong or the reconstruction is wrong.
        # Actually, there are many variants.
        # Let's use the one from the "Strassen algorithm" Wikipedia page or standard textbooks.
        
        # Variant:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = (A00 - A10)(B10 + B11)
        # M4 = A00B00
        # M5 = A11B11
        # M6 = (A01 - A11)(B00 + B01)
        # M7 = (A00 + A01)(B10 - B00)
        
        # C00 = M1 + M4 - M5 + M6
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M5
        
        # It seems M7 is always excluded in this specific set?
        # That can't be right. Rank 7 means 7 independent products.
        # If M7 is not used in the sum, it's wasted?
        # Or maybe M7 is used in a different variant.
        
        # Let's try a different variant where all 7 are used.
        # Or maybe I can just use the schoolbook for the inner 2x2x2 (rank 8).
        # Then total rank for 4x4x4 is 7 * 8 = 56.
        # Total rank for 4x4x8 is 56 * 2 = 112.
        # 112 is still better than 128.
        # And it's much easier to implement correctly.
        
        # Let's implement the "Block Strassen" (rank 7 outer, rank 8 inner) scheme.
        # Outer: 7 products of 2x2 matrices (blocks).
        # Inner: Schoolbook multiplication of 2x2 matrices (rank 8).
        
        # For each of the 7 outer products:
        # We have A_comb (2x2 matrix of blocks) and B_comb (2x2 matrix of blocks).
        # We compute P = A_comb * B_comb.
        # P is a 2x2 matrix of blocks.
        # We use schoolbook for 2x2x2.
        # P[i,j] = sum_k A_comb[i,k] * B_comb[k,j].
        # This gives 8 terms (2x2x2).
        # Each term is a product of two 2x2 matrices (blocks).
        # These are the "scalar" multiplications.
        # We add these to the global U, V, W.
        
        # Mapping indices:
        # Outer loop: prod in 0..6
        # Inner loop: i2, j2, k2 in 0..1 (for 2x2x2 schoolbook)
        # Total 7 * 8 = 56 columns per 4x4x4 subproblem.
        
        for prod in range(7):
            # A_comb and B_comb are 2x2 matrices of coefficients for the blocks.
            # A_comb[I,J] is the coefficient for A_block[I,J] in the "row I, col J" of the 2x2 matrix A_s.
            # Wait, A_s is a 2x2 matrix.
            # A_s[I,J] = sum_{I',J'} A_coefs[prod][I',J'] * A_block[I',J']?
            # No, that's not right.
            # A_s is formed by linear combinations of the 4 blocks.
            # A_s[0,0] = sum A_coefs[prod][0,0] * A_block[0,0]?
            # If A_coefs is [[1,0],[0,1]], then A_s[0,0]=A00, A_s[1,1]=A11.
            # So A_s[I,J] = A_coefs[prod][I,J] * A_block[I,J].
            # This assumes the linear combination is "diagonal" in the block sense?
            # No, A_coefs[prod] is a 2x2 matrix.
            # So A_s[I,J] is a linear combination of ALL blocks?
            # No, A_coefs[prod][I,J] is a scalar.
            # So A_s[I,J] = A_coefs[prod][I,J] * A_block[I,J].
            # This means A_s has the same 2x2 block structure as A, but scaled?
            # For M1: A_s = diag(A00, A11).
            # For M2: A_s = [[0,0],[A10,A11]].
            # This is correct for the "block" level.
            
            # Now we multiply A_s (2x2) by B_s (2x2).
            # P_s[I,J] = sum_K A_s[I,K] * B_s[K,J].
            # A_s[I,K] = A_coefs[prod][I,K] * A_block[I,K].
            # B_s[K,J] = B_coefs[prod][K,J] * B_block[K,J].
            # So P_s[I,J] = sum_K A_coefs[prod][I,K] B_coefs[prod][K,J] A_block[I,K] B_block[K,J].
            
            # This is a sum of products of blocks.
            # We add this to the global decomposition.
            # For each I, J, K:
            # Term: (A_coefs[prod][I,K] * A_block[I,K]) * (B_coefs[prod][K,J] * B_block[K,J])
            # This contributes to P_s[I,J].
            # P_s[I,J] is a 2x2 matrix (block of C).
            # It is added to the reconstruction of C.
            
            # But we are decomposing the tensor.
            # The "scalar" multiplication is A_block[I,K] * B_block[K,J].
            # We need to decompose this 2x2x2 multiplication.
            # We use schoolbook for this.
            
            for I in range(2):
                for K in range(2):
                    a_coeff = A_coefs[prod][I][K]
                    if a_coeff == 0: continue
                    for J in range(2):
                        b_coeff = B_coefs[prod][K][J]
                        if b_coeff == 0: continue
                        # The product is A_block[I,K] * B_block[K,J].
                        # We decompose this 2x2x2 multiplication using schoolbook.
                        # A_block[I,K] is a 2x2 matrix. B_block[K,J] is a 2x2 matrix.
                        # C_sub[I,K][r,s] * B_block[K,J][s,t] -> C_sub[I,J][r,t].
                        # Indices in 4x4:
                        # A_block[I,K] is A[I*2:I*2+2, K*2:K*2+2].
                        # B_block[K,J] is B[K*2:K*2+2, J*2:J*2+2].
                        # C_block[I,J] is C[I*2:I*2+2, J*2:J*2+2].
                        
                        # Schoolbook for 2x2x2:
                        # C[r,t] = sum_s A[r,s] B[s,t].
                        # 8 terms: r, s, t in 0..1.
                        
                        # Global column index for this term:
                        # offset + prod * 8 + (I*2 + K*2 + J*2 + r*2 + s*2 + t) ?
                        # No, we need to assign a unique column for each term.
                        # There are 7 (prod) * 2 (I) * 2 (K) * 2 (J) * 8 (inner terms) = 448 terms?
                        # No, many coefficients are 0.
                        # But we should allocate 7 * 8 = 56 columns if we use Strassen outer and schoolbook inner?
                        # Wait.
                        # The product A_s * B_s is a 2x2 matrix.
                        # We computed it as sum_{I,K,J} ...
                        # This is NOT a direct tensor decomposition.
                        # We are summing multiple block products to form P_s.
                        # Each block product A_block[I,K] * B_block[K,J] is a 2x2x2 tensor multiplication.
                        # We decompose each one using schoolbook (rank 8).
                        # So for each (I,K,J) with non-zero coefficients, we add 8 columns.
                        
                        # How many (I,K,J) pairs are there?
                        # I in 0..1, K in 0..1, J in 0..1. 8 pairs.
                        # For each prod, we have up to 8 pairs.
                        # For each pair, we add 8 columns.
                        # So 7 prods * 8 pairs * 8 terms = 448 columns?
                        # That's way more than 56.
                        
                        # I think I am confusing the levels.
                        # Let's restart the decomposition logic.
                        
                        pass

    # Given the complexity and risk of error, I will fall back to the schoolbook rank 128 solution.
    # It is guaranteed to be correct and matches the seed score.
    
    rank = 128
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
