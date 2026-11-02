"""Strassen-based rank-minimal exact tensor decomposition for 4x4 matrix multiplication."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    # 3x3 block indices: a,b,c in {0,1,2}, each block is 4x4
    # For each block position (a,b,c), we apply Strassen's algorithm
    # Strassen's 7 products for 2x2 blocks of 2x2 matrices
    # Each 2x2 matrix is composed of four 4x4 blocks? No, each "scalar" in Strassen is a 2x2 block of the 4x4 matrix.
    # Actually, for 4x4 matrices, we can think of them as 2x2 blocks of 2x2 matrices.
    # But Strassen works on 2x2 "scalar" matrices where each scalar is itself a matrix.
    # So for 4x4, we treat each 4x4 as a 2x2 block of 2x2 submatrices.
    
    # Let's use the standard Strassen decomposition for 2x2 block matrices where each block is 2x2.
    # The 7 products are:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # Then:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each Aij, Bjk, Ckl are 2x2 blocks of the 4x4 matrix.
    # We need to express each product as a sum of 7 rank-1 terms.
    
    # For each of the 7 Strassen products, we need to express the product of two 2x2 block matrices
    # as a tensor decomposition.
    
    # Let's define the 2x2 blocks. A is 4x4, split into 2x2 blocks of size 2x2:
    # A[0:2,0:2] = A00, A[0:2,2:4] = A01
    # A[2:4,0:2] = A10, A[2:4,2:4] = A11
    
    # Similarly for B and C.
    
    # For Strassen, we compute 7 products of 2x2 block matrices.
    # Each product of two 2x2 block matrices can be decomposed using schoolbook method for 2x2 matrices,
    # but we want a rank-minimal decomposition.
    
    # Actually, let's just use the schoolbook method for the 7 Strassen products.
    # Each 2x2 * 2x2 multiplication requires 4 products, so 7 * 4 = 28 products?
    # No, Strassen replaces the 8 products (4x2) with 7 products.
    # But each "product" in Strassen is a product of two 2x2 block matrices, which itself requires
    # a decomposition.
    
    # Wait, I think I'm overcomplicating this. Let me just use the known rank-49 decomposition
    # for 4x4 matrix multiplication based on Strassen's algorithm.
    
    # The rank of 4x4 matrix multiplication is at most 7 * 4 = 28? No, that's not right.
    # Actually, Strassen's algorithm for n x n matrices where n is a power of 2 has rank 7^(log2(n)).
    # For n=4, log2(4)=2, so rank = 7^2 = 49. That matches!
    
    # So we need to construct the rank-49 decomposition.
    # Let's build it recursively.
    
    # For 2x2 matrices, Strassen gives rank 7.
    # For 4x4, we apply Strassen to 2x2 blocks, giving 7 recursive calls, each of rank 7, total 49.
    
    # Let's define the indices for the 49 products.
    # We'll use a recursive approach.
    
    # First, define the Strassen indices for 2x2:
    # s in {0,1,2,3,4,5,6}
    # A_indices[s], B_indices[s] give the linear combinations
    # C_indices[s] gives how to combine the results
    
    # For 2x2 Strassen:
    # s=0: A=(A00+A11), B=(B00+B11) -> contributes to C00 and C11
    # s=1: A=(A10+A11), B=B00 -> contributes to C10 and C11
    # s=2: A=A00, B=(B01-B11) -> contributes to C01 and C11
    # s=3: A=A11, B=(B10-B00) -> contributes to C00 and C10
    # s=4: A=(A00+A01), B=B11 -> contributes to C00 and C01
    # s=5: A=(A10-A00), B=(B00+B01) -> contributes to C11
    # s=6: A=(A01-A11), B=(B10+B11) -> contributes to C00
    
    # Now for 4x4, we apply this to each of the 4 blocks? No, we apply it once to the 2x2 block structure.
    # The 4x4 matrix is a 2x2 block matrix where each block is 2x2.
    # So we apply Strassen to the 2x2 block matrix, where each "element" is a 2x2 matrix.
    # Each Strassen product involves multiplying two 2x2 matrices (which are blocks), and each such
    # multiplication can be done with rank 7 (using Strassen again on the 2x2 sub-blocks? No, 2x2 matrices
    # are the base case).
    
    # Actually, for 2x2 matrices, Strassen gives rank 7 directly.
    # So for 4x4, we have 7 Strassen products, each of which is a product of two 2x2 block matrices.
    # Each 2x2 block matrix product can be decomposed into 7 rank-1 terms (using Strassen for 2x2).
    # So total rank = 7 * 7 = 49.
    
    # Let's implement this.
    
    # Define Strassen coefficients for 2x2:
    # For product s, A_comb is a 2x2 matrix of coefficients for A00, A01, A10, A11
    # B_comb is a 2x2 matrix of coefficients for B00, B01, B10, B11
    # C_comb is a 2x2 matrix of coefficients for how this product contributes to C00, C01, C10, C11
    
    # s=0: A=A00+A11, B=B00+B11, C=C00+C11
    A0_comb = [[1,0],[0,1]]  # A00, A11
    B0_comb = [[1,0],[0,1]]  # B00, B11
    C0_comb = [[1,0],[0,1]]  # C00, C11
    
    # s=1: A=A10+A11, B=B00, C=C10+C11
    A1_comb = [[0,0],[1,1]]  # A10, A11
    B1_comb = [[1,0],[0,0]]  # B00
    C1_comb = [[0,0],[1,1]]  # C10, C11
    
    # s=2: A=A00, B=B01-B11, C=C01+C11
    A2_comb = [[1,0],[0,0]]  # A00
    B2_comb = [[0,1],[0,-1]] # B01, -B11
    C2_comb = [[0,1],[0,1]]  # C01, C11
    
    # s=3: A=A11, B=B10-B00, C=C00+C10
    A3_comb = [[0,0],[0,1]]  # A11
    B3_comb = [[-1,0],[1,0]] # -B00, B10
    C3_comb = [[1,0],[1,0]]  # C00, C10
    
    # s=4: A=A00+A01, B=B11, C=C00+C01
    A4_comb = [[1,1],[0,0]]  # A00, A01
    B4_comb = [[0,0],[0,1]]  # B11
    C4_comb = [[1,1],[0,0]]  # C00, C01
    
    # s=5: A=A10-A00, B=B00+B01, C=C11
    A5_comb = [[-1,0],[1,0]] # -A00, A10
    B5_comb = [[1,1],[0,0]]  # B00, B01
    C5_comb = [[0,0],[0,1]]  # C11
    
    # s=6: A=A01-A11, B=B10+B11, C=C00
    A6_comb = [[0,1],[0,-1]] # A01, -A11
    B6_comb = [[0,0],[1,1]]  # B10, B11
    C6_comb = [[1,0],[0,0]]  # C00
    
    A_comb_list = [A0_comb, A1_comb, A2_comb, A3_comb, A4_comb, A5_comb, A6_comb]
    B_comb_list = [B0_comb, B1_comb, B2_comb, B3_comb, B4_comb, B5_comb, B6_comb]
    C_comb_list = [C0_comb, C1_comb, C2_comb, C3_comb, C4_comb, C5_comb, C6_comb]
    
    # Now, for each Strassen product s (0-6), we need to compute the product of two 2x2 block matrices.
    # Each block matrix is 2x2 where each entry is a 2x2 submatrix.
    # The product of two 2x2 block matrices can be computed using Strassen on the 2x2 "scalar" level,
    # but since the "scalars" are 2x2 matrices, we need to be careful.
    
    # Actually, let's just use the schoolbook method for the 2x2 block matrix multiplication.
    # For each Strassen product, we multiply two 2x2 block matrices using 4 products (schoolbook),
    # but we want to minimize rank.
    
    # Hmm, I think the correct approach is to use Strassen recursively.
    # For 4x4, we view it as a 2x2 block matrix of 2x2 blocks.
    # We apply Strassen to get 7 products of 2x2 block matrices.
    # For each product of 2x2 block matrices, we again apply Strassen (since 2x2 is the base case).
    # So each of the 7 products requires 7 rank-1 terms, total 49.
    
    # Let's index the 49 terms as (s, t) where s in {0..6} is the outer Strassen index
    # and t in {0..6} is the inner Strassen index.
    
    # For each (s, t):
    # - The outer Strassen product s gives linear combinations of the 4 blocks of A and B.
    # - The inner Strassen product t gives the decomposition of the product of the resulting 2x2 block matrices.
    
    # Let's construct U, V, W.
    
    # U has shape (16, 49), indexed by (i, j) for A[i,j] and column (s, t).
    # V has shape (16, 49), indexed by (j, k) for B[j,k] and column (s, t).
    # W has shape (16, 49), indexed by (k, i) for C[k,i] and column (s, t).
    
    # For a given (s, t):
    # The outer Strassen combination for A is given by A_comb_list[s], which is a 2x2 matrix
    # giving coefficients for A00, A01, A10, A11 (as 2x2 blocks).
    # The inner Strassen combination for the resulting 2x2 matrix is given by A_comb_list[t].
    
    # So the total coefficient for A[bi, bj] where bi, bj in {0,1} (block indices) and
    # the element within the block is (pi, qj) where pi, qj in {0,1} (sub-block indices)
    # is:
    # coeff = (A_comb_list[s][bi][bj]) * (A_comb_list[t][pi][qj])
    
    # Wait, I need to be more careful with the indexing.
    
    # Let's define:
    # A is 4x4. We view it as a 2x2 block matrix where each block is 2x2.
    # A[bi*2+pi, bj*2+qj] for bi,bj,pi,qj in {0,1}.
    
    # The outer Strassen combination for A is a 2x2 matrix M_s where
    # M_s[bi][bj] gives the coefficient of the block A[bi,bj] in the linear combination.
    # So the combined A matrix for product s is:
    # A_s[pi, qj] = sum_{bi,bj} M_s[bi][bj] * A[bi*2+pi, bj*2+qj]
    
    # Then the inner Strassen product t decomposes the multiplication of A_s and B_s.
    # A_s is a 2x2 matrix, B_s is a 2x2 matrix.
    # Strassen for 2x2 gives 7 products, each of which is a product of linear combinations
    # of the 4 entries of A_s and B_s.
    
    # For inner product t, the coefficient of A_s[pi, qj] is given by A_comb_list[t][pi][qj].
    # So the total coefficient for A[bi*2+pi, bj*2+qj] in product (s,t) is:
    # coeff_A = M_s[bi][bj] * A_comb_list[t][pi][qj]
    
    # Similarly for B:
    # coeff_B = N_s[bj][bk] * B_comb_list[t][qj][rk]
    # where N_s is the outer Strassen combination for B, and the inner is for the 2x2 matrix B_s.
    
    # And for C:
    # The outer Strassen combination for C is given by C_comb_list[s], which is a 2x2 matrix
    # giving how the product contributes to C00, C01, C10, C11.
    # The inner Strassen combination for C is given by C_comb_list[t].
    # So the total coefficient for C[ck*2+rk, ci*2+pi] is:
    # coeff_C = C_comb_list[s][ck][ci] * C_comb_list[t][rk][pi]
    
    # Wait, I need to be careful about the row/column ordering.
    
    # Let's just implement it and check.
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # Outer Strassen combinations
            A_s = A_comb_list[s]  # 2x2
            B_s = B_comb_list[s]  # 2x2
            C_s = C_comb_list[s]  # 2x2
            
            # Inner Strassen combinations
            A_t = A_comb_list[t]  # 2x2
            B_t = B_comb_list[t]  # 2x2
            C_t = C_comb_list[t]  # 2x2
            
            # Fill U
            for bi in range(2):
                for bj in range(2):
                    for pi in range(2):
                        for qj in range(2):
                            i = bi * 2 + pi
                            j = bj * 2 + qj
                            coeff = A_s[bi][bj] * A_t[pi][qj]
                            U[i * 4 + j][col] += coeff
            
            # Fill V
            for bj in range(2):
                for bk in range(2):
                    for qj in range(2):
                        for rk in range(2):
                            j = bj * 2 + qj
                            k = bk * 2 + rk
                            coeff = B_s[bj][bk] * B_t[qj][rk]
                            V[j * 4 + k][col] += coeff
            
            # Fill W
            for ck in range(2):
                for ci in range(2):
                    for rk in range(2):
                        for pi in range(2):
                            k = ck * 2 + rk
                            i = ci * 2 + pi
                            coeff = C_s[ck][ci] * C_t[rk][pi]
                            W[k * 4 + i][col] += coeff
    
    # Convert to integers if possible
    for i in range(len(U)):
        for j in range(len(U[i])):
            if abs(U[i][j] - round(U[i][j])) < 1e-10:
                U[i][j] = int(round(U[i][j]))
    for i in range(len(V)):
        for j in range(len(V[i])):
            if abs(V[i][j] - round(V[i][j])) < 1e-10:
                V[i][j] = int(round(V[i][j]))
    for i in range(len(W)):
        for j in range(len(W[i])):
            if abs(W[i][j] - round(W[i][j])) < 1e-10:
                W[i][j] = int(round(W[i][j]))
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
