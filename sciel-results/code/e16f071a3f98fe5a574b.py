"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    # We use a block-level Kronecker structure:
    # The 4x4 * 4x6 multiplication can be decomposed into 4x4 blocks (top-left, top-right, bottom-left, bottom-right)
    # but since p=6, the second matrix has shape 4x6, so we can think of it as two 4x3 blocks horizontally.
    # Actually, let's think of A as 4x4, B as 4x6. We can partition A into 2x2 blocks of size 2x2,
    # and B into 2x3 blocks of size 2x2 and 2x4? No, that's not clean.
    #
    # Let's use the known result: for 4x4 * 4x6, the rank is at most 77 (from the knowledge base).
    # The knowledge says: "Replacing the per-element 2x2x2x3 decomposition (88 rank) with a block-level 
    # Kronecker structure: 11 outer terms (7 Strassen + 4 schoolbook) each expanded via rank-7 inner 
    # Strassen gives 77 rank, a 12.5% reduction"
    #
    # So the strategy is:
    # 1. Partition the problem into blocks where we can apply Strassen.
    # 2. Use 7 Strassen multiplications + 4 schoolbook for the outer structure (11 total).
    # 3. Each inner multiplication uses rank-7 Strassen for 2x2 * 2x2 (but adapted for the sizes).
    #
    # Let me think more carefully. A is 4x4, B is 4x6, C is 4x6.
    # We can partition A into four 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # We can partition B into two 2x3 blocks vertically? No, B is 4x6.
    # B = [[B11, B12, B13], [B21, B22, B23]] where each Bij is 2x2? No, 2x2 + 2x2 + 2x2 = 2x6. Yes!
    # So B has 6 columns, we can split into three 2-column blocks.
    #
    # Actually, let's partition differently. Let's use 2x2 blocks for A (4x4 = 2x2 blocks of 2x2).
    # For B (4x6), we can partition into 2x2 blocks: 4 rows = 2 blocks of 2 rows, 6 cols = 3 blocks of 2 cols.
    # So B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x2.
    #
    # C = A*B = [[C11, C12, C13], [C21, C22, C23]] where each is 2x2.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    #
    # This is 12 multiplications of 2x2 * 2x2, each of which has rank 7 (Strassen).
    # Total rank = 12 * 7 = 84. But the knowledge says 77.
    #
    # Wait, the knowledge says "11 outer terms (7 Strassen + 4 schoolbook)". 
    # Let me think about how to get 11 outer terms.
    #
    # Maybe we can use Strassen at the block level too. Let's partition A into 2x2 blocks of 2x2,
    # and B into 2x3 blocks... hmm.
    #
    # Actually, let me just implement the 12*7 = 84 rank decomposition first, and then see if I can 
    # optimize to 77. But the knowledge says 77 was measured, so let me try to understand it.
    #
    # Alternative: maybe we partition A as 4x4 into 2x2 blocks, and B as 4x6 into 2x2 and 2x4 blocks?
    # No, let's stick with uniform 2x2 blocks.
    #
    # Let me just implement the straightforward 12 * 7 = 84 rank decomposition using Strassen for each 2x2*2x2.
    # Then I'll see if I can find a better decomposition.
    #
    # Actually, let me re-read: "11 outer terms (7 Strassen + 4 schoolbook) each expanded via rank-7 inner Strassen gives 77 rank"
    # 11 * 7 = 77. So there are 11 outer terms, each of rank 7.
    #
    # How do we get 11 outer terms? 
    # For 4x4 * 4x6, if we think of it as multiplying a 4x4 by a 4x6, we can write:
    # C = A * B
    # 
    # One approach: use the fact that 4x4 * 4x4 has rank 49 (Strassen-based), and then handle the extra 2 columns.
    # 4x4 * 4x4: rank 49. Then 4x4 * 4x2: rank ? 
    # Total would be 49 + rank(4x4 * 4x2).
    #
    # For 4x4 * 4x2: partition into 2x2 blocks. A has 4 blocks, B has 2 blocks (vertically).
    # C1 = A11*B1 + A12*B2, C2 = A21*B1 + A22*B2. That's 4 multiplications of 2x2*2x2, rank 4*7=28.
    # Total: 49 + 28 = 77. That matches!
    #
    # So the decomposition is:
    # - Split B into B_left (4x4) and B_right (4x2).
    # - C_left = A * B_left (4x4 * 4x4), rank 49.
    # - C_right = A * B_right (4x4 * 4x2), rank 28.
    # - Total rank = 49 + 28 = 77.
    #
    # Now I need to implement this. Let me build the U, V, W matrices.
    #
    # For the 4x4 * 4x4 part (rank 49):
    # Use Strassen's algorithm recursively. 4x4 * 4x4 can be done with 7 multiplications of 2x2*2x2,
    # each of rank 7. So 7 * 7 = 49.
    #
    # For the 4x4 * 4x2 part (rank 28):
    # 4 multiplications of 2x2*2x2, each of rank 7. So 4 * 7 = 28.
    #
    # Let me implement this step by step.
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (24, rank)
    # W shape: (p*n, rank) = (24, rank)
    
    # We'll build the decomposition in two parts:
    # Part 1: A * B_left where B_left is 4x4 (columns 0-3)
    # Part 2: A * B_right where B_right is 4x2 (columns 4-5)
    
    # Helper: create a 2x2 * 2x2 Strassen decomposition
    # Returns lists of (u_idx, v_idx, w_idx, coeff) for each of the 7 rank-1 terms
    def strassen_2x2():
        # Strassen's 7 multiplications for 2x2 * 2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each term is a rank-1 contribution to the tensor.
        # We need to express each Mj as a sum of rank-1 terms.
        # For 2x2 * 2x2, each Mj = X * Y where X is 2x2 and Y is 2x2.
        # But in the tensor decomposition, we need rank-1 terms.
        # 
        # Actually, for the tensor decomposition of 2x2 * 2x2, each of the 7 Strassen multiplications
        # corresponds to a rank-1 term in the tensor. Wait, no. Each Mj is a product of two 2x2 matrices,
        # which is itself a bilinear form. The tensor for 2x2*2x2 has rank 7, and each of the 7 terms
        # in Strassen is a rank-1 tensor.
        
        # Let me think about this differently. The matrix multiplication tensor T for 2x2*2x2 is:
        # T[i,j,k,l] = 1 if i+k=j+l (using 0-indexed, i,k for A, j,l for B, result C[i,j])
        # Wait, the convention is W[k*n+i, r] for the output.
        
        # Let me use a direct approach. For 2x2 * 2x2:
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = A*B
        # C[0,0] = a00*b00 + a01*b10
        # C[0,1] = a00*b01 + a01*b11
        # C[1,0] = a10*b00 + a11*b10
        # C[1,1] = a10*b01 + a11*b11
        
        # The tensor T has indices (i, j, k) where i is row of A, j is col of B, k is... 
        # Actually, the standard tensor decomposition for matrix multiplication C = A*B is:
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # The tensor T[i,j,k] = 1 (it's the multiplication tensor).
        # A rank-r decomposition is: T[i,j,k] = sum_r U[i,r] * V[k,r] * W[j,r]
        # Wait, the convention here is W[k*n+i, r] which is a bit different.
        
        # Let me just build the decomposition directly using the known Strassen formulas.
        # For 2x2 * 2x2, I'll return 7 terms, each specifying which elements of A, B, C are involved.
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # This contributes to C00, C01, C10, C11
        # As a rank-1 term in the tensor: 
        # U part: A00 + A11 -> indices (0,0) and (1,1) in A
        # V part: B00 + B11 -> indices (0,0) and (1,1) in B  
        # W part: contributes to C with coefficients
        
        # This is getting complicated. Let me use a different approach.
        # I'll build the full 4x4 * 4x6 decomposition directly using known results.
        
        # Actually, let me just use the schoolbook decomposition for now and verify it works,
        # then try to optimize.
        pass
    
    # Let me use a simpler approach: build the decomposition using numpy to verify correctness,
    # then extract the integer/half-integer entries.
    
    # For now, let's use the schoolbook decomposition (rank 96) as a fallback,
    # and try to implement the 77-rank decomposition.
    
    # Actually, let me implement the 77-rank decomposition properly.
    # 
    # Strategy:
    # 1. Split B into B_left (4x4) and B_right (4x2)
    # 2. C_left = A * B_left using rank-49 decomposition
    # 3. C_right = A * B_right using rank-28 decomposition
    # 4. Total rank = 77
    
    # For step 2: 4x4 * 4x4 with rank 49
    # Use Strassen at the block level: partition into 2x2 blocks
    # A = [[A11, A12], [A21, A22]], B_left = [[B11, B12], [B21, B22]]
    # C_left = [[C11, C12], [C21, C22]]
    # 
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mj is a 2x2 * 2x2 multiplication, which has rank 7.
    # So total rank for 4x4*4x4 is 7 * 7 = 49.
    
    # For step 3: 4x4 * 4x2 with rank 28
    # Partition: A = [[A11, A12], [A21, A22]], B_right = [[B1], [B2]] where B1, B2 are 2x2
    # C_right = [[C1], [C2]] where C1 = A11*B1 + A12*B2, C2 = A21*B1 + A22*B2
    # 4 multiplications of 2x2*2x2, each rank 7, total 28.
    
    # Now I need to implement this. Let me create helper functions.
    
    # First, let me define the 2x2 * 2x2 rank-7 decomposition.
    # I'll represent each rank-1 term as (a_indices, b_indices, c_indices, coeffs)
    
    # For 2x2 * 2x2, the 7 Strassen terms are:
    # Each term is of the form: sum_{i,j} a_ij * sum_{k,l} b_kl * sum_{m,n} c_mn
    # where a_ij, b_kl, c_mn are 0 or 1 (or -1 for some terms).
    
    # Let me define the 7 terms explicitly.
    # Term 1: (A00+A11)(B00+B11) -> contributes to C00(+), C01(+), C10(+), C11(-)
    # Wait, I need to be more careful.
    
    # Let me use a different representation. For each of the 7 Strassen multiplications,
    # I'll define:
    # - The linear combination of A elements
    # - The linear combination of B elements  
    # - The contribution to C elements
    
    # Actually, the cleanest way is to just build the U, V, W matrices directly.
    
    # Let me define a function that, given a 2x2*2x2 Strassen term, returns the
    # contributions to U, V, W for a specific block position.
    
    # This is getting complex. Let me just use numpy to construct the decomposition
    # and then extract the entries.
    
    import numpy as np
    
    # Build the full 4x4 * 4x6 decomposition
    # We'll create U, V, W as numpy arrays and fill them in.
    
    rank = 77
    U = np.zeros((16, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    # Part 1: A * B_left (4x4 * 4x4), rank 49
    # B_left uses columns 0-3 of B
    # A uses all 16 elements (4x4)
    # C_left uses columns 0-3 of C (4x4), which in W's indexing is k*n+i for k=0..3, i=0..3
    
    # Strassen for 4x4 * 4x4:
    # A blocks: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    # B blocks: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    # C blocks: C11=C[0:2,0:2], C12=C[0:2,2:4], C21=C[2:4,0:2], C22=C[2:4,2:4]
    
    # For each Strassen term, we need a 2x2*2x2 decomposition.
    # Let me define the 2x2*2x2 rank-7 decomposition.
    
    # For 2x2 * 2x2, using Strassen:
    # The 7 multiplications are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mj is a scalar product of two linear combinations.
    # In the tensor decomposition, each Mj corresponds to a rank-1 term.
    
    # For M1: (A00+A11)(B00+B11)
    # U part: A00 + A11 -> U[0, term] = 1, U[3, term] = 1 (if A is indexed row-major)
    # Wait, for a 2x2 block at position (block_row, block_col) in the 4x4 matrix,
    # the element A[i,j] in the block corresponds to A_global[block_row*2+i, block_col*2+j]
    # In U's indexing, the row is i*m + j where i is the row of A and j is the column of A.
    # So for A[0,0] in the global 4x4, U row = 0*4 + 0 = 0
    # For A[0,1], U row = 0*4 + 1 = 1
    # For A[1,0], U row = 1*4 + 0 = 4
    # For A[1,1], U row = 1*4 + 1 = 5
    # etc.
    
    # This is getting very detailed. Let me just write out the 2x2*2x2 decomposition
    # as a list of 7 terms, each specifying the U, V, W contributions.
    
    # For a 2x2 * 2x2 multiplication at a specific position:
    # A is at rows [ar, ar+1], cols [ac, ac+1] in the global A
    # B is at rows [br, br+1], cols [bc, bc+1] in the global B
    # C is at rows [cr, cr+1], cols [cc, cc+1] in the global C
    
    # U index for A[i,j] = i*4 + j
    # V index for B[i,j] = i*6 + j
    # W index for C[i,j] = j*4 + i  (using the convention W[k*n+i, r] where k is col, i is row)
    
    # Let me define a function to add a 2x2*2x2 Strassen decomposition.
    
    def add_strassen_2x2(U, V, W, ar, ac, br, bc, cr, cc, offset):
        # ar, ac: top-left of A block
        # br, bc: top-left of B block
        # cr, cc: top-left of C block
        # offset: column offset in U, V, W
        
        # 7 Strassen terms
        # M1 = (A00+A11)(B00+B11)
        # U: A[ar,ac] + A[ar+1,ac+1]
        U[ar*4 + ac, offset] += 1
        U[(ar+1)*4 + (ac+1), offset] += 1
        # V: B[br,bc] + B[br+1,bc+1]
        V[br*6 + bc, offset] += 1
        V[(br+1)*6 + (bc+1), offset] += 1
        # W: contributes to C[cr,cc](+), C[cr,cc+1](+), C[cr+1,cc](+), C[cr+1,cc+1](-)
        W[cc*4 + cr, offset] += 1       # C[cr,cc]
        W[(cc+1)*4 + cr, offset] += 1   # C[cr,cc+1]
        W[cc*4 + (cr+1), offset] += 1   # C[cr+1,cc]
        W[(cc+1)*4 + (cr+1), offset] -= 1  # C[cr+1,cc+1]
        
        # M2 = (A10+A11)B00
        U[ar*4 + ac, offset+1] += 0  # placeholder
        U[(ar+1)*4 + ac, offset+1] += 1
        U[(ar+1)*4 + (ac+1), offset+1] += 1
        V[br*6 + bc, offset+1] += 1
        # W: contributes to C[cr+1,cc](+), C[cr+1,cc+1](-)
        W[cc*4 + (cr+1), offset+1] += 1
        W[(cc+1)*4 + (cr+1), offset+1] -= 1
        
        # M3 = A00(B10-B11)
        U[ar*4 + ac, offset+2] += 1
        V[(br+1)*6 + bc, offset+2] += 1
        V[(br+1)*6 + (bc+1), offset+2] -= 1
        # W: contributes to C[cr,cc+1](+), C[cr+1,cc+1](+)
        W[(cc+1)*4 + cr, offset+2] += 1
        W[(cc+1)*4 + (cr+1), offset+2] += 1
        
        # M4 = A11(B01-B00)
        U[(ar+1)*4 + (ac+1), offset+3] += 1
        V[br*6 + (bc+1), offset+3] += 1
        V[br*6 + bc, offset+3] -= 1
        # W: contributes to C[cr,cc](+), C[cr+1,cc](+)
        W[cc*4 + cr, offset+3] += 1
        W[cc*4 + (cr+1), offset+3] += 1
        
        # M5 = (A00+A01)B11
        U[ar*4 + ac, offset+4] += 1
        U[ar*4 + (ac+1), offset+4] += 1
        V[(br+1)*6 + (bc+1), offset+4] += 1
        # W: contributes to C[cr,cc](-), C[cr+1,cc+1](+)
        W[cc*4 + cr, offset+4] -= 1
        W[(cc+1)*4 + (cr+1), offset+4] += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(ar+1)*4 + ac, offset+5] += 1
        U[ar*4 + ac, offset+5] -= 1
        V[br*6 + bc, offset+5] += 1
        V[br*6 + (bc+1), offset+5] += 1
        # W: contributes to C[cr+1,cc+1](+)
        W[(cc+1)*4 + (cr+1), offset+5] += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[ar*4 + (ac+1), offset+6] += 1
        U[(ar+1)*4 + (ac+1), offset+6] -= 1
        V[(br+1)*6 + bc, offset+6] += 1
        V[(br+1)*6 + (bc+1), offset+6] += 1
        # W: contributes to C[cr,cc](+), C[cr+1,cc](+)
        W[cc*4 + cr, offset+6] += 1
        W[cc*4 + (cr+1), offset+6] += 1
        
        return offset + 7
    
    # Now build Part 1: 4x4 * 4x4
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each of these is a 2x2*2x2 multiplication.
    # A11 is at (0,0), A12 at (0,2), A21 at (2,0), A22 at (2,2)
    # B11 at (0,0), B12 at (0,2), B21 at (2,0), B22 at (2,2)
    # C11 at (0,0), C12 at (0,2), C21 at (2,0), C22 at (2,2)
    
    # M1 = (A11+A22)(B11+B22)
    # This is a product of two sums, so it's not a single 2x2*2x2.
    # I need to expand it. (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # That's 4 separate 2x2*2x2 multiplications.
    #
    # Wait, that's not right. In Strassen's algorithm, M1 is computed as a single multiplication
    # of two 2x2 matrices: X = A11+A22, Y = B11+B22, then M1 = X*Y.
    # X*Y is a 2x2*2x2 multiplication, which has rank 7.
    # So M1 contributes 7 rank-1 terms.
    #
    # But X = A11+A22 is a 2x2 matrix, and Y = B11+B22 is a 2x2 matrix.
    # So I need to compute X*Y using the 2x2*2x2 decomposition.
    #
    # The issue is that X and Y are linear combinations of the original A and B blocks.
    # So when I apply the 2x2*2x2 decomposition to X*Y, the U part will involve
    # the elements of X (which are sums of A11 and A22 elements), and similarly for V.
    
    # Let me redefine add_strassen_2x2 to take explicit U and V coefficient vectors.
    
    def add_term(U, V, W, u_coeffs, v_coeffs, w_coeffs, offset):
        # u_coeffs: dict mapping U row index to coefficient
        # v_coeffs: dict mapping V row index to coefficient
        # w_coeffs: dict mapping W row index to coefficient
        for idx, coeff in u_coeffs.items():
            U[idx, offset] += coeff
        for idx, coeff in v_coeffs.items():
            V[idx, offset] += coeff
        for idx, coeff in w_coeffs.items():
            W[idx, offset] += coeff
        return offset + 1
    
    # This is getting too complex. Let me just use the schoolbook decomposition for now
    # and return it. The rank will be 96, which is worse than 77, but at least it's correct.
    #
    # Actually, let me try a different approach. Let me just hardcode the 77-rank decomposition
    # by constructing it carefully.
    
    # For now, let's return the schoolbook decomposition.
    rank_schoolbook = n * m * p  # 96
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
