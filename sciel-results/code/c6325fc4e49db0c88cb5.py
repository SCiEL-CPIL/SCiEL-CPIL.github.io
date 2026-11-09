"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Block-level Strassen decomposition:
    # Treat the 2x4x8 multiplication as a 2x2 multiplication of sub-blocks.
    # A is 2x4, B is 4x8, C is 2x8.
    # Split A into two 2x2 blocks: A = [A1 | A2] where A1 = A[:, 0:2], A2 = A[:, 2:4]
    # Split B into two 2x4 blocks: B = [B1; B2] where B1 = B[0:2, :], B2 = B[2:4, :]
    # Then C = A1 * B1 + A2 * B2
    # 
    # Each product A1*B1 is 2x2 * 2x4 = 2x4, and A2*B2 is 2x2 * 2x4 = 2x4.
    # 
    # Now apply Strassen to the 2x2 block multiplication of the "matrix" of 2x2 blocks:
    # M1 = A1, M2 = A2, N1 = B1, N2 = B2
    # But B1 and B2 are 2x4, not square. We need to handle this carefully.
    #
    # Actually, let's think of it differently. The full multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    #
    # Group j into j1 in {0,1} and j2 in {2,3}.
    # C[i,k] = sum_{j1 in {0,1}} A[i,j1]*B[j1,k] + sum_{j2 in {0,1}} A[i,j2+2]*B[j2+2,k]
    #
    # Let A1[i,j1] = A[i,j1] for i in {0,1}, j1 in {0,1}  (2x2)
    # Let A2[i,j2] = A[i,j2+2] for i in {0,1}, j2 in {0,1}  (2x2)
    # Let B1[j1,k] = B[j1,k] for j1 in {0,1}, k in {0,...,7}  (2x8)
    # Let B2[j2,k] = B[j2+2,k] for j2 in {0,1}, k in {0,...,7}  (2x8)
    #
    # C = A1 * B1 + A2 * B2 where A1,A2 are 2x2 and B1,B2 are 2x8.
    #
    # Now, for each of the two products (A1*B1 and A2*B2), we have a 2x2 * 2x8 product.
    # We can further split B into 4x2 blocks: B1 = [B1a | B1b] where B1a = B1[:, 0:4], B1b = B1[:, 4:8]
    # Similarly B2 = [B2a | B2b].
    #
    # So C = A1*B1a + A1*B1b + A2*B2a + A2*B2b
    # Each term is 2x2 * 2x4 = 2x4.
    #
    # For a 2x2 * 2x4 product, we can use Strassen-like decomposition.
    # Actually, for 2x2 * 2xn, the standard approach gives rank 4*2 = 8 per product? 
    # No, let me think about this more carefully.
    #
    # For a 2x2 * 2x4 matrix product, the schoolbook rank is 2*2*4 = 16.
    # We can do better by grouping columns. Let's use rank 7 for 2x2*2x2 (Strassen) 
    # and apply it to each 2x2x2 sub-block, giving 4 sub-blocks * 7 = 28 per 2x2*2x4.
    # That's worse than 16, so that's not helpful.
    #
    # Alternative: For 2x2 * 2x4, we can write it as two 2x2 * 2x2 products side by side.
    # Each 2x2*2x2 has Strassen rank 7, so total rank 14 per 2x2*2x4.
    # With 4 such products, total rank = 4*14 = 56.
    #
    # This is better than schoolbook 64!
    
    # Strassen decomposition for 2x2 * 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_2x2(a, b, base_col):
        """Apply Strassen to 2x2 * 2x2, returning rank-1 term contributions.
        a is 2x2, b is 2x2, base_col is the starting column index.
        Returns list of (U_idx, V_idx, W_idx, coeff) tuples."""
        terms = []
        
        # Define the 7 Strassen products
        # Each product is a sum of products of linear combinations of a and b entries
        # For rank-1 terms, we need to express each M as a single product
        
        # M1 = (A11 + A22)(B11 + B22)
        # This is not a single product unless we expand. But in tensor decomposition,
        # each "product" of sums requires multiple rank-1 terms.
        # Actually, in Strassen's algorithm, each M_i is computed as a single matrix product
        # of two 2x2 matrices (which are linear combinations of the original entries).
        # But here we're decomposing into rank-1 terms of the original tensor.
        
        # Let me reconsider. The Strassen algorithm computes 7 products of 2x2 matrices,
        # where each 2x2 matrix is a linear combination of the original 2x2 matrices.
        # Each 2x2*2x2 product has schoolbook rank 4, so 7*4 = 28 rank-1 terms total.
        # That's worse than the schoolbook 4 for a single 2x2*2x2!
        
        # I think I'm confusing myself. Let me restart with a clearer approach.
        
        # The key insight: for 2x2 * 2x4, we can view it as computing two 2x2*2x2 products.
        # If we use schoolbook for each 2x2*2x2, that's 4+4 = 8 rank-1 terms per 2x2*2x4.
        # With 4 such sub-products, total = 4*8 = 32. That's the same as schoolbook!
        
        # To get improvement, we need to use Strassen properly.
        # Strassen reduces 2x2*2x2 from 4 to 7... wait, 7 > 4. That's worse!
        # 
        # Oh wait, I had it backwards. Strassen uses 7 multiplications instead of 8 for 
        # 2x2*2x2? No, schoolbook for 2x2*2x2 uses 4 multiplications (one per output entry),
        # and Strassen uses 7. So Strassen is WORSE for 2x2!
        # 
        # Strassen is better for 2x2*2x2 in terms of arithmetic operations when considering
        # the recursive structure, but for a single level, 7 > 4.
        #
        # I think I've been confused. Let me just use schoolbook with rank 64 and verify it works.
        return terms
    
    # Schoolbook rank 64 decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
