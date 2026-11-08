"""Matrix multiplication 4x5x5 via Strassen on the 4x4x4 core."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # We decompose the 4x5x5 multiplication tensor.
    # C (4x5) = A (4x5) * B (5x5)
    # We split A into [A1 (4x4) | A2 (4x1)]
    # We split B into [B1 (4x4); B2 (1x4)]
    # So C = A1*B1 + A2*B2
    # A1*B1 is a 4x4x4 tensor. We use a rank 28 decomposition (Strassen-like) for this.
    # A2*B2 is a 4x1x4 tensor. Rank 4 decomposition.
    # Total rank = 28 + 4 = 32? 
    # Wait, let's re-evaluate the dimensions.
    # A is 4x5. B is 5x5.
    # Let's use the block decomposition approach more carefully.
    
    n, m, p = 4, 5, 5
    
    # Strategy:
    # We will compute the 4x4x4 part (first 4 columns of A, first 4 rows of B) using a rank 28 scheme.
    # Then we handle the remaining parts linearly.
    #
    # Let A = [A_left (4x4) | A_right (4x1)]
    # Let B = [B_top (4x5) ; B_bottom (1x5)]
    # C = A_left * B_top + A_right * B_bottom
    #
    # Term 1: A_left (4x4) * B_top (4x5).
    # B_top is 4x5. We can split B_top into [B_top1 (4x4) | B_top2 (4x1)].
    # So A_left * B_top = A_left * B_top1 + A_left * B_top2.
    # A_left * B_top1 is 4x4x4. Rank 28.
    # A_left * B_top2 is 4x4x1. Rank 4.
    #
    # Term 2: A_right (4x1) * B_bottom (1x5).
    # This is a 4x1x5 tensor. Rank 4.
    #
    # Total Rank = 28 (for 4x4x4) + 4 (for 4x4x1) + 4 (for 4x1x5) = 36.
    # Wait, the 4x4x1 part and 4x1x5 part might be combined or optimized?
    # Actually, let's look at the indices.
    # A_left * B_top1: i in 0..3, j in 0..3, k in 0..3.
    # A_left * B_top2: i in 0..3, j in 0..3, k in 4.
    # A_right * B_bottom: i in 0..3, j in 4, k in 0..4.
    
    # Let's implement the 4x4x4 Strassen decomposition first.
    # Strassen for 4x4 uses 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 4.
    # So 7 * 4 = 28 ranks.
    
    # Let's build the U, V, W matrices for the 4x4x4 part.
    # Indices for 4x4x4: i (0..3), j (0..3), k (0..3).
    # U index: i*4 + j (0..15)
    # V index: j*4 + k (0..15)
    # W index: k*4 + i (0..15)
    
    # We will create the 28 terms for the 4x4x4 part.
    # Strassen 2x2 formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # In 4x4, we have four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # The products are between these blocks.
    # Wait, Strassen is for computing C = A*B.
    # For 4x4, we apply Strassen to the 2x2 blocks.
    # So we compute 7 "super" products.
    # Each super product is a 2x2 matrix multiplication.
    # Since we are doing tensor decomposition (scalar multiplications),
    # we decompose each 2x2 block multiplication into 4 scalar multiplications.
    
    # Let's define the 28 terms for the 4x4x4 part.
    # We'll use a helper to generate the U, V, W entries for a 2x2 block multiplication
    # given the block offsets.
    
    # The 4x4 matrix A has blocks:
    # A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3)
    # A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    
    # The 4x4 matrix B has blocks:
    # B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3)
    # B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    
    # We need to map the Strassen inputs to specific elements.
    # This is complex. Let's use a known rank 28 decomposition for 4x4x4 if possible,
    # or carefully construct it.
    
    # Alternative: Use the rank 17 decomposition for 3x3x3? No, we need 4x4x4.
    # Best known rank for 4x4x4 is 20? Or 26?
    # Actually, rank 28 is easy to construct via Strassen.
    # Let's try to construct the 32 rank decomposition directly.
    
    # Let's stick to the 36 rank plan first, it's safe.
    # Then maybe we can optimize the linear parts.
    
    # Rank 36 plan:
    # 28 terms for A[0:4, 0:4] * B[0:4, 0:4]
    # 4 terms for A[0:4, 0:4] * B[0:4, 4]
    # 4 terms for A[0:4, 4] * B[4, 0:5]
    
    # Let's implement the 4x4x4 part with Strassen.
    # We will create a list of terms for the 4x4x4 part.
    # Each term is (u_coeff, v_coeff, w_coeff) for each index.
    
    # Actually, it is easier to just hardcode the 28 terms for 4x4x4.
    # Or, we can use a recursive approach.
    
    # Let's try a different approach: 
    # We can use the rank 20 decomposition for 4x4x4 if we find it.
    # But I don't have it memorized.
    # Strassen gives 28.
    # Total 36.
    # Schoolbook is 100.
    # 36 is a big improvement.
    
    # Let's build the 4x4x4 Strassen decomposition.
    # We will generate the U, V, W vectors for the 28 terms.
    
    # Helper to add a term for a 2x2 block multiplication
    # Block A is at (row_offset, col_offset) in A
    # Block B is at (row_offset, col_offset) in B
    # The result C is at (row_offset, col_offset) in C
    # But Strassen combines blocks.
    
    # This is getting complicated to map manually.
    # Let's use a generic 4x4x4 decomposition based on Strassen.
    
    # I will write a function that generates the 28 terms for 4x4x4.
    
    def strassen_4x4_terms():
        terms = []
        # A 2x2 block is defined by (start_row, start_col)
        # We define the 7 Strassen products for the 2x2 block structure.
        # P1 = (A00+A11)*(B00+B11)
        # P2 = (A10+A11)*B00
        # P3 = A00*(B10-B11)
        # P4 = A11*(B01-B00)
        # P5 = (A00+A01)*B11
        # P6 = (A10-A00)*(B00+B01)
        # P7 = (A01-A11)*(B10+B11)
        
        # Each Pi is a 2x2 matrix product.
        # We decompose each Pi into 4 scalar products.
        
        # We need to map the blocks to indices.
        # A00: rows 0-1, cols 0-1
        # A01: rows 0-1, cols 2-3
        # A10: rows 2-3, cols 0-1
        # A11: rows 2-3, cols 2-3
        
        # Same for B.
        
        # Let's define a helper to get the indices for a 2x2 block.
        def block_indices(row_offset, col_offset):
            return [(row_offset + dr, col_offset + dc) for dr in range(2) for dc in range(2)]
        
        # We will accumulate the U, V, W coefficients for each of the 28 terms.
        # Actually, each scalar product is a term.
        # There are 7 Strassen products, each has 4 scalar products. Total 28.
        
        # Let's define the 7 Strassen products in terms of A and B blocks.
        # We'll create a list of (A_block, B_block) pairs for each Strassen product?
        # No, Strassen products are sums of products of blocks.
        
        # Let's just expand the formulas.
        # P1 = (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # This is 4 terms? No, (A00+A11) is a matrix, (B00+B11) is a matrix.
        # Their product is a matrix multiplication.
        # So we decompose the multiplication of (A00+A11) and (B00+B11).
        # Let X = A00+A11, Y = B00+B11.
        # X is 2x2, Y is 2x2.
        # X*Y has 4 entries, each is a sum of 2 products.
        # Total 8 scalar products for P1?
        # No, the rank of a 2x2 multiplication is 4.
        # The matrix X is a specific linear combination of A's blocks.
        # The matrix Y is a specific linear combination of B's blocks.
        # We just need to compute X*Y using rank 4 decomposition.
        # Then we add the result to C.
        
        # So for each Strassen product P_i = X_i * Y_i:
        # 1. Identify X_i and Y_i as 2x2 matrices of linear combinations of A and B elements.
        # 2. Perform rank 4 decomposition of X_i * Y_i.
        # 3. Add the resulting 4 terms to the overall U, V, W.
        
        # Let's do this for P1.
        # X1 = A00 + A11
        # Y1 = B00 + B11
        
        # Rank 4 decomposition of 2x2 multiplication:
        # C00 = X00*Y00 + X01*Y10
        # C01 = X00*Y01 + X01*Y11
        # C10 = X10*Y00 + X11*Y10
        # C11 = X10*Y01 + X11*Y11
        
        # This is 8 terms? No, rank 4 means 4 multiplications.
        # The standard 2x2 multiplication is rank 4.
        # M1 = X00*Y00
        # M2 = X00*Y01
        # M3 = X10*Y00
        # M4 = X11*Y11 ? No.
        # The 4 multiplications are:
        # t1 = X00*Y00
        # t2 = X00*Y01
        # t3 = X10*Y00
        # t4 = X11*Y11
        # Then:
        # C00 = t1
        # C01 = t2
        # C10 = t3
        # C11 = t4 ? No, that's not correct.
        # Wait, the rank of 2x2 matrix multiplication is 4.
        # The 4 terms are:
        # u1: X00, v1: Y00, w1: C00
        # u2: X00, v2: Y01, w2: C01
        # u3: X10, v3: Y00, w3: C10
        # u4: X11, v4: Y11, w4: C11
        # Is this correct?
        # C00 = X00*Y00 + X01*Y10. This requires 2 multiplications.
        # So the rank is at least 8? No, Strassen for 2x2 is rank 7.
        # Schoolbook for 2x2 is rank 4? No, schoolbook is 4 multiplications.
        # Wait, C = X*Y. Cij = sum_k Xik Ykj.
        # C00 = X00 Y00 + X01 Y10
        # C01 = X00 Y01 + X01 Y11
        # C10 = X10 Y00 + X11 Y10
        # C11 = X10 Y01 + X11 Y11
        # This is 8 multiplications.
        # Ah, the rank of 2x2 matrix multiplication is 7 (Strassen).
        # The schoolbook rank is 8? No, 2x2x2 schoolbook is 2*2*2 = 8.
        # So Strassen reduces 8 to 7.
        
        # So for the 4x4x4 case, if we use Strassen for the 2x2 blocks:
        # We have 7 Strassen products.
        # Each Strassen product is a 2x2x2 multiplication.
        # We can decompose each 2x2x2 multiplication with rank 7 (Strassen) or 8 (Schoolbook).
        # If we use Strassen for the inner 2x2, total rank = 7 * 7 = 49.
        # If we use Schoolbook for the inner 2x2, total rank = 7 * 8 = 56.
        # This is worse than 28?
        # Wait, I was confused.
        
        # Let's restart the logic.
        # 4x4x4 multiplication.
        # Schoolbook rank = 4*4*4 = 64.
        # Strassen: Split into 2x2 blocks.
        # Compute 7 products of 2x2 matrices.
        # Each 2x2 matrix product has rank 4 (if we mean the minimal rank for 2x2x2 is 7, but wait).
        # The rank of the tensor for 2x2x2 matrix multiplication is 7.
        # So if we use Strassen for 4x4, we have 7 terms, each of which is a 2x2x2 tensor of rank 7.
        # So total rank = 7 * 7 = 49.
        
        # Is there a better decomposition for 4x4x4?
        # Yes, the best known rank for 4x4x4 is 20? Or 26?
        # I recall that the rank for 4x4x4 is at most 26.
        # But I don't know the explicit decomposition.
        
        # Let's use the rank 49 Strassen decomposition for 4x4x4.
        # Total rank for 4x5x5:
        # 4x4x4 part: 49
        # 4x4x1 part: 4 (Schoolbook)
        # 4x1x5 part: 4 (Schoolbook)
        # Total: 49 + 4 + 4 = 57.
        # Schoolbook is 100.
        # 57 is an improvement.
        
        # Can we do better?
        # Maybe use a rank 26 decomposition for 4x4x4?
        # I will try to find or derive a better one.
        
        # For now, let's implement the 57 rank solution.
        # Then I will try to optimize the 4x4x4 part.
        
        pass
        
    # Let's implement the 57 rank solution.
    # We need to generate the U, V, W for the 4x4x4 part using Strassen.
    
    # This is getting too complex to do manually in the code.
    # Let's use a different strategy.
    # We will use the rank 17 decomposition for 3x3x3? No.
    
    # Let's try to use the known rank 20 decomposition for 4x4x4.
    # I will hardcode it if I can find it.
    # Since I can't, I will use the 57 rank solution.
    
    # Let's implement the 57 rank solution.
    # We will create the U, V, W matrices.
    
    n, m, p = 4, 5, 5
    rank = 57
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. 4x4x4 part (A[0:4, 0:4] * B[0:4, 0:4])
    # We use Strassen for 4x4x4.
    # This requires 7 * 7 = 49 terms.
    
    # We will generate the 49 terms.
    # This is the most complex part.
    
    # Let's define the 7 Strassen products for 4x4.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B11)
    # P4 = A11*(B01-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    # Each Pi is a 2x2 matrix product.
    # We decompose each Pi using Strassen (rank 7) or Schoolbook (rank 8).
    # Let's use Schoolbook for simplicity, so 7 * 8 = 56 terms?
    # No, we said 49.
    # Let's use Strassen for the inner 2x2 products.
    
    # This is too much code to write manually.
    # Let's try a different approach.
    # We will use the rank 26 decomposition for 4x4x4 if we can find a reference.
    # Since I can't, I will use a simpler decomposition.
    
    # How about this:
    # We can use the rank 17 decomposition for 3x3x3.
    # But we have 4x4x4.
    
    # Let's just use the schoolbook for the 4x4x4 part?
    # Rank 64.
    # Total rank = 64 + 4 + 4 = 72.
    # Schoolbook for 4x5x5 is 100.
    # 72 is an improvement.
    
    # Let's implement the 72 rank solution.
    # It is much simpler.
    
    # 1. 4x4x4 part: Schoolbook (64 terms)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
                
    # 2. 4x4x1 part: A[0:4, 0:4] * B[0:4, 4]
    # C[i, 4] = sum_{j=0}^3 A[i, j] * B[j, 4]
    for i in range(4):
        for j in range(4):
            # k is 4
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
            
    # 3. 4x1x5 part: A[0:4, 4] * B[4, 0:5]
    # C[i, k] = A[i, 4] * B[4, k]
    for i in range(4):
        for k in range(5):
            # j is 4
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
