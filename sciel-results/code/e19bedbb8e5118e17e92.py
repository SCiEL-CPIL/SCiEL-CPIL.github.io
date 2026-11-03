import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # We will use a decomposition based on Strassen for 4x4x4 
    # with schoolbook 2x2 leaves, applied to two 4x4 blocks of B.
    # Rank of 4x4x4 with this method is 28.
    # Total rank = 28 + 28 = 56.
    
    # First, generate the Strassen decomposition for 4x4x4 with schoolbook 2x2 leaves.
    # This returns U4, V4, W4 of size 16x28, 16x28, 16x28.
    # Indices: U4[i*4+j], V4[j*4+k], W4[k*4+i]
    
    def strassen_4x4_decomp():
        # Schoolbook decomposition for 2x2x2
        # Returns U2, V2, W2 of size 4x4, 4x4, 4x4
        # U2[i*2+j], V2[j*2+k], W2[k*2+i]
        def schoolbook_2x2():
            U2 = [[0]*4 for _ in range(4)]
            V2 = [[0]*4 for _ in range(4)]
            W2 = [[0]*4 for _ in range(4)]
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U2[i*2+j][col] = 1
                        V2[j*2+k][col] = 1
                        W2[k*2+i][col] = 1
                        col += 1
            return U2, V2, W2

        U2, V2, W2 = schoolbook_2x2()
        
        # Strassen's algorithm for 4x4x4
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # We need to construct the tensor decomposition.
        # Each Pm is a 2x2x2 multiplication.
        # We have 7 such multiplications.
        # Each has rank 4 (schoolbook).
        # Total rank 28.
        
        U4 = [[0]*28 for _ in range(16)]
        V4 = [[0]*28 for _ in range(16)]
        W4 = [[0]*28 for _ in range(16)]
        
        # Mapping of Strassen terms to 2x2 blocks
        # Term m uses U2, V2, W2 with specific signs and block offsets.
        
        # Helper to add a 2x2 decomposition to the 4x4 arrays
        # offset_u: row offset in A (0 or 2)
        # offset_v: col offset in A / row offset in B (0 or 2)
        # offset_w: col offset in B / row offset in C (0 or 2)
        # sign_u, sign_v, sign_w: multipliers for the blocks
        
        def add_term(m, sign_u1, sign_u2, sign_v1, sign_v2, sign_w1, sign_w2, col_offset):
            # This is complex to generalize with signs for each element.
            # Instead, we explicitly build the 28 terms.
            pass

        # Let's build the 28 terms explicitly for the 7 Strassen products.
        # Each product Pm is a sum of 4 schoolbook terms.
        
        # P1 = (A11+A22)(B11+B22)
        # This is (A11 B11) + (A11 B22) + (A22 B11) + (A22 B22)
        # Wait, Strassen combines them.
        # The decomposition is linear.
        # P1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        # Each of these is a 2x2 multiplication.
        # A11B11: U is A11, V is B11.
        # A11B22: U is A11, V is B22.
        # A22B11: U is A22, V is B11.
        # A22B22: U is A22, V is B22.
        
        # We can just iterate over the 4 components of each Strassen product.
        
        # Define the 7 Strassen products as sums of 4 basic 2x2 products.
        # Product 1: A11B11, A11B22, A22B11, A22B22
        # Product 2: A21B11, A22B11
        # Product 3: A11B12, -A11B22
        # Product 4: A22B21, -A22B11
        # Product 5: A11B22, A12B22
        # Product 6: A21B11, A21B12, -A11B11, -A11B12
        # Product 7: A12B21, A12B22, -A22B21, -A22B22
        
        # Each "AiBj" is a 2x2x2 multiplication.
        # We decompose each into 4 schoolbook terms.
        
        # List of (A_block, B_block, sign)
        # A_block: (row_offset, col_offset) in 4x4 A
        # B_block: (row_offset, col_offset) in 4x4 B
        # Note: A is 4x4, B is 4x4.
        # A11: (0,0), A12: (0,2), A21: (2,0), A22: (2,2)
        # B11: (0,0), B12: (0,2), B21: (2,0), B22: (2,2)
        
        strassen_terms = [
            # P1
            ((0,0), (0,0), 1), ((0,0), (2,0), 1), ((2,0), (0,0), 1), ((2,0), (2,0), 1),
            # P2
            ((2,0), (0,0), 1), ((2,2), (0,0), 1),
            # P3
            ((0,0), (0,2), 1), ((0,0), (2,2), -1),
            # P4
            ((2,2), (2,0), 1), ((2,2), (0,0), -1),
            # P5
            ((0,0), (2,2), 1), ((0,2), (2,2), 1),
            # P6
            ((2,0), (0,0), 1), ((2,0), (0,2), 1), ((0,0), (0,0), -1), ((0,0), (0,2), -1),
            # P7
            ((0,2), (2,0), 1), ((0,2), (2,2), 1), ((2,2), (2,0), -1), ((2,2), (2,2), -1)
        ]
        
        # Total 28 terms.
        for idx, (a_off, b_off, sign) in enumerate(strassen_terms):
            a_r0, a_c0 = a_off
            b_r0, b_c0 = b_off
            
            # Add 4 schoolbook terms for this 2x2 product
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # Global indices in 4x4
                        gi = a_r0 + i
                        gj = a_c0 + j
                        gk = b_r0 + k
                        # Wait, B is 4x4. The inner index of A matches outer of B.
                        # A is 4x4, B is 4x4.
                        # A[i,j] * B[j,k].
                        # In our 2x2 block, A_block is rows a_r0..a_r0+1, cols a_c0..a_c0+1.
                        # B_block is rows b_r0..b_r0+1, cols b_c0..b_c0+1.
                        # For the product to be valid, the col of A_block must match row of B_block.
                        # In Strassen, A11 (cols 0-1) multiplies B11 (rows 0-1). Match.
                        # A11 (cols 0-1) multiplies B22 (rows 2-3). Mismatch!
                        # Strassen works because the blocks are aligned in the 4x4 grid.
                        # A11 is top-left. B11 is top-left.
                        # A11 B11 contributes to C11.
                        # A11 B22 contributes to C12? No.
                        # A11 is A[0:2, 0:2]. B22 is B[2:4, 2:4].
                        # A11 B22 is not a standard block multiplication in the 4x4 product.
                        # Strassen's algorithm uses sums/differences of blocks.
                        # The "multiplication" is of the combined blocks.
                        # (A11+A22) is a 2x2 matrix. (B11+B22) is a 2x2 matrix.
                        # Their product is a 2x2 matrix.
                        # So we are multiplying two 2x2 matrices.
                        # The decomposition of this 2x2 multiplication is standard.
                        # The result is added to the appropriate block of C.
                        
                        # My list above decomposes the Strassen products into basic 2x2 multiplications.
                        # But A11 B22 is NOT a term in the 4x4 multiplication.
                        # The term is (A11+A22)(B11+B22).
                        # This expands to A11B11 + A11B22 + A22B11 + A22B22.
                        # These are 2x2 matrix multiplications.
                        # A11 is 2x2, B22 is 2x2.
                        # The product A11B22 is a 2x2 matrix.
                        # This product contributes to the C11 block?
                        # No, (A11+A22)(B11+B22) contributes to C11 and C12 and C21 and C22?
                        # No, it's one of the 7 terms.
                        # P1 contributes to C11, C12, C21, C22 with specific coefficients.
                        
                        # This approach of expanding into basic 2x2 multiplications is valid for the tensor decomposition.
                        # The tensor for 4x4x4 is a sum of 28 rank-1 tensors.
                        # Each rank-1 tensor corresponds to a scalar multiplication in the expanded Strassen algorithm.
                        
                        # Let's verify the indices.
                        # Term: A11 B22.
                        # A11 indices: i in 0-1, j in 0-1.
                        # B22 indices: j in 2-3, k in 2-3.
                        # Wait, B22 rows are 2-3. A11 cols are 0-1.
                        # They don't match!
                        # In Strassen, the "multiplication" is abstract.
                        # The algorithm computes P1 = (A11+A22)(B11+B22).
                        # This is a 2x2 matrix multiplication.
                        # The inputs are the sum of blocks.
                        # The decomposition of this 2x2 multiplication is standard.
                        # The U matrix for this term will have non-zeros in the A11 and A22 blocks.
                        # The V matrix for this term will have non-zeros in the B11 and B22 blocks.
                        # The W matrix will have non-zeros in the C blocks.
                        
                        # So, for each of the 7 Strassen products, we use the 2x2 Strassen decomposition (or schoolbook).
                        # Since we are using schoolbook for the 2x2 leaves, we just need to map the indices.
                        
                        # Let's restart the term generation.
                        # We have 7 products P1..P7.
                        # Each Pm is a 2x2 matrix multiplication.
                        # We decompose each Pm using schoolbook (4 terms).
                        # Total 28 terms.
                        
                        # For each term in Pm:
                        # It involves specific elements of the "combined" A and B blocks.
                        # E.g., P1 term (i,j,k) involves A_comb[i,j] and B_comb[j,k].
                        # A_comb = A11+A22. So A_comb[i,j] = A11[i,j] + A22[i,j].
                        # B_comb = B11+B22. So B_comb[j,k] = B11[j,k] + B22[j,k].
                        # The rank-1 tensor for this scalar multiplication has:
                        # U: non-zeros at A11[i,j] and A22[i,j].
                        # V: non-zeros at B11[j,k] and B22[j,k].
                        # W: non-zeros at the C blocks where P1 contributes.
                        
                        # This is getting complicated to hardcode.
                        # I will use a simpler known decomposition.
                        # Rank of 4x4x4 is at most 20? No.
                        # I will stick to the schoolbook 96 if I can't easily get 56.
                        # But 56 is a big win.
                        
                        # Let's try a different structural change: 
                        # Use the fact that 3x4x8 can be decomposed into 
                        # 3x4x4 + 3x4x4.
                        # Rank of 3x4x4 is at most 3*4*4 = 48.
                        # Total 96.
                        # Can we do better for 3x4x4?
                        # 3x4x4 is close to 4x4x4.
                        # If we use Strassen for 4x4x4 (rank 28), and pad 3x4 to 4x4,
                        # the rank is 28.
                        # So 3x4x4 has rank at most 28.
                        # Then 3x4x8 has rank at most 28 + 28 = 56.
                        
                        # I will implement the 4x4x4 Strassen decomposition with schoolbook leaves.
                        # I will hardcode the 28 terms.
                        
                        break
                    break
                break
            break
        # Due to complexity, I will return the schoolbook 96 for now to ensure validity.
        # But I will try to implement the 56 rank.
        
        # Let's use a library or precomputed values.
        # Since I cannot use external libs for the decomposition, I will construct it.
        
        # For the sake of this step, I will return the schoolbook decomposition.
        rank = n * m * p
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

    return strassen_4x4_decomp()
# EVOLVE-BLOCK-END
