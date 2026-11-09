"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # A is n x m, B is m x p, C is n x p
    # Tensor indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # So C is indexed by (k, i) where k in 0..p-1, i in 0..n-1
    # This means C has shape (p, n) in the tensor's native indexing, but
    # the task says "output factor follows W[k*n+i,r]" so we just follow that.

    # Strategy: Use Strassen for the 4x4x4 part (rank 7) for the first 4 columns,
    # and schoolbook for the remaining 3 columns (rank 12).
    # Total rank = 7 + 12 = 19? No, Strassen is 7 rank-1 terms for 4x4x4.
    # Wait: Strassen's algorithm for 4x4 matrices uses 7 multiplications of 2x2 blocks.
    # But each 2x2 multiplication itself has rank 7 (schoolbook for 2x2 is rank 4, 
    # but Strassen for 2x2 is rank 7).
    
    # Actually, let me reconsider. The standard Strassen algorithm for multiplying
    # two 4x4 matrices: divide into 2x2 blocks, use 7 products of 2x2 blocks.
    # Each 2x2 product can be done in rank 4 (schoolbook) or rank 7 (Strassen on 2x2).
    # If we use Strassen recursively: rank(4x4x4) = 7 * rank(2x2x2) = 7*4 = 28.
    # That's the same as schoolbook (16*... no, schoolbook for 4x4x4 is 4*4*4=64? No.)
    
    # Wait, I'm confusing myself. Let me think about rank more carefully.
    # Schoolbook for m x n x p: rank = m*n*p.
    # For 4x4x4: schoolbook rank = 4*4*4 = 64.
    # Strassen for 4x4x4: The standard Strassen algorithm for 4x4 uses 7 products of 2x2.
    # Each 2x2x2 schoolbook has rank 4. So total rank = 7*4 = 28.
    # But actually, the rank of matrix multiplication tensor for 4x4x4 is known to be 
    # at most 49 using Strassen (if you use Strassen recursively: 7^2 = 49).
    # No wait: Strassen(2x2x2) = 7. Strassen(4x4x4) = 7 * Strassen(2x2x2) = 7*7 = 49.
    
    # Hmm, but schoolbook 4x4x4 = 64. So Strassen gives 49, which is better.
    
    # For the 4x4x3 part (last 3 columns): schoolbook rank = 4*4*3 = 48.
    # Can we do better? The (2x2)x(2x3) subproblem... 
    # Actually, let's just use schoolbook for the 4x4x3 part: rank 48.
    # Total: 49 + 48 = 97.
    
    # Alternative: Use Strassen(4x4x4) = 49 for first 4 cols, schoolbook(4x4x3) = 48 for last 3 cols.
    # Total = 97.
    
    # From the inspiration: "Replaces schoolbook rank-48 for the 4x4x3 subproblem with a 
    # rank-44 decomposition using Strassen(7)+schoolbook(4) per (2x2)*(2x3) block, reducing 
    # total rank from 97 to 93"
    
    # Let me construct this:
    # For 4x4x3: Split A into 2x2 blocks (4 blocks), B into 2x2 blocks (3 blocks: 2 full 2x2 and one 2x1... 
    # no, 4 rows, 3 cols: split into 2 row-blocks of 2, and 2 col-blocks where first is 2 wide, second is 1 wide).
    # Actually, split B (4x3) into 2 row-blocks (each 2x3) and then each 2x3 into (2x2) + (2x1).
    
    # This is getting complex. Let me just implement the straightforward approach:
    # Strassen for 4x4x4 (rank 49) + schoolbook for 4x4x3 (rank 48) = 97.
    # Or try to get the 93 rank version.
    
    # Let me implement Strassen for 4x4x4 first.
    
    def strassen_4x4_4x4():
        """Return U, V, W factors (as lists of lists) for 4x4x4 Strassen decomposition.
        Rank 49.
        U: 16 x 49, V: 16 x 49, W: 16 x 49
        """
        # Strassen for 4x4: divide into 2x2 blocks.
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # 7 products of 2x2 matrices.
        # Each 2x2x2 schoolbook has rank 4.
        # So we have 7 * 4 = 28 rank-1 terms? No, that's not right either.
        
        # Actually, Strassen's algorithm computes:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # Each Mi is a 2x2 matrix product, which in schoolbook has 4 rank-1 terms.
        # So total rank = 7 * 4 = 28? But the known rank of 4x4x4 tensor is at most 49.
        # 28 seems too low. The issue is that Strassen's 7 products are of 2x2 matrices,
        # and each 2x2 multiplication tensor has rank 4. So 7*4 = 28 rank-1 terms for the 4x4x4?
        # That can't be right because the rank is known to be at least 23 and at most 49.
        
        # I think the confusion is: Strassen gives an upper bound on the rank. 
        # For 2x2x2, Strassen gives rank 7 (but schoolbook is rank 4, so Strassen is worse for 2x2).
        # For 4x4x4, using Strassen: 7 products of 2x2 matrices. If each 2x2 product uses 
        # schoolbook (rank 4), total rank = 7*4 = 28. But this contradicts the known upper bound of 49.
        
        # Wait, I think I'm wrong. Let me reconsider. The rank of the 2x2x2 tensor is exactly 4 
        # (it's bilinear, so rank = 2*2*2 = 4 for schoolbook, and it's known that rank(2x2x2) = 4).
        # So Strassen's 7 products of 2x2 matrices, each with rank 4, gives total rank 7*4 = 28.
        # But this would mean rank(4x4x4) <= 28, which I don't think is correct.
        
        # The issue: Strassen's algorithm for 4x4 doesn't directly give 28 rank-1 terms because
        # the 7 "products" are not independent rank-1 decompositions. The additions and subtractions
        # create dependencies.
        
        # Let me just use a different approach. I'll use the known Strassen decomposition 
        # for 4x4x4 which has rank 49 (7^2).
        
        # Actually, for simplicity and correctness, let me just use schoolbook for everything:
        # Rank = 4*4*7 = 112. That's what the seed program does.
        
        # To improve, let me try a hybrid approach:
        # Use the fact that we can decompose the 4x4x7 tensor into a 4x4x4 part and a 4x4x3 part.
        # For 4x4x4, use a rank-49 Strassen-like decomposition.
        # For 4x4x3, use schoolbook (rank 48) or better.
        
        # But implementing Strassen correctly is tricky. Let me try a simpler improvement:
        # Use rank-1 terms more efficiently by exploiting structure.
        
        pass
    
    # Let me just implement a correct rank-93 decomposition based on the inspiration.
    # The key insight from the cell findings: "Strassen(7)+schoolbook(4) per (2x2)*(2x3) block"
    
    # For the 4x4x7 problem:
    # Split into 4x4x4 (first 4 cols) + 4x4x3 (last 3 cols)
    
    # For 4x4x4: Use rank-49 Strassen decomposition
    # For 4x4x3: Use rank-44 decomposition
    
    # Let me construct the Strassen 4x4x4 decomposition.
    # The 4x4x4 Strassen decomposition with rank 49 comes from applying Strassen 
    # (rank 7 for 2x2x2) to each of the 7 products in the top-level Strassen.
    # So it's a recursive Strassen: rank = 7^2 = 49.
    
    # Here's how: 
    # Level 1: 7 products of 2x2 matrices.
    # Level 2: Each 2x2 product is done via Strassen (rank 7 for 2x2x2).
    # Total: 7 * 7 = 49 rank-1 terms.
    
    # Let me define the Strassen 2x2x2 decomposition first.
    
    def strassen_2x2_2x2():
        """Strassen decomposition for 2x2x2. Rank 7.
        U: 4x7, V: 4x7, W: 4x7
        A is 2x2: indices 0,1,2,3 = (0,0),(0,1),(1,0),(1,1)
        B is 2x2: same
        C is 2x2: same
        """
        # Strassen's 7 products:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 - M4 + M5 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 + M7
        
        # For each Mi, it's a product of two linear forms in A and two linear forms in B,
        # contributing one rank-1 term.
        
        # U[i,j] for A[i,j], V[i,j] for B[i,j], W[i,j] for C[i,j]
        # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # M1: (A00+A11)*(B00+B11)
        # U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C11=1, C01=1, C10=1? 
        # No. M1 contributes to C00, C11. 
        # C00 = M1 - M4 + M5 + M6
        # C11 = M1 - M3 + M2 + M7
        # So M1 has coefficient +1 in C00 and +1 in C11.
        
        # Each rank-1 term: U_col, V_col, W_col
        # Term 1 (M1):
        U[0][0] = 1; U[3][0] = 1  # A00, A11
        V[0][0] = 1; V[3][0] = 1  # B00, B11
        W[0][0] = 1; W[3][0] = 1  # C00, C11
        
        # Term 2 (M2):
        U[2][1] = 1; U[3][1] = 1  # A10, A11
        V[0][1] = 1               # B00
        W[2][1] = 1; W[3][1] = 1  # C10, C11
        
        # Term 3 (M3):
        U[0][2] = 1               # A00
        V[1][2] = 1; V[3][2] = -1 # B01, -B11
        W[0][2] = -1; W[1][2] = 1 # -C00, C01
        # Wait, M3 = a00*(b01-b11)
        # C01 = M3 + M5, so M3 has +1 in C01
        # C00 = M1 - M4 + M5 + M6, M3 not in C00
        # C11 = M1 - M3 + M2 + M7, so M3 has -1 in C11
        W[0][2] = 0; W[1][2] = 1; W[3][2] = -1  # C01, -C11
        
        # Term 4 (M4):
        U[3][3] = 1               # A11
        V[2][3] = 1; V[0][3] = -1 # B10, -B00
        W[0][3] = -1; W[2][3] = 1 # -C00, C10
        
        # Term 5 (M5):
        U[0][4] = 1; U[1][4] = 1 # A00, A01
        V[3][4] = 1               # B11
        W[0][4] = 1; W[1][4] = 1  # C00, C01
        
        # Term 6 (M6):
        U[2][5] = 1; U[0][5] = -1 # A10, -A00
        V[0][5] = 1; V[1][5] = 1  # B00, B01
        W[0][5] = 1; W[2][5] = 1  # C00, C10
        
        # Term 7 (M7):
        U[1][6] = 1; U[3][6] = -1 # A01, -A11
        V[2][6] = 1; V[3][6] = 1  # B10, B11
        W[3][6] = 1               # C11
        
        return U, V, W
    
    # Now for 4x4x4 Strassen: apply Strassen at 2x2 block level.
    # This is complex. Let me just use schoolbook for now and see if I can 
    # at least beat 112.
    
    # Actually, let me try a simpler structural improvement:
    # Use the fact that the tensor for 4x4x7 can be seen as 7 copies of 4x4x1,
    # but that's not helpful.
    
    # Let me just implement the schoolbook but verify it's correct, and see
    # if there's a simple way to reduce rank.
    
    # For now, let me implement the 4x4x4 Strassen (rank 49) + 4x4x3 schoolbook (rank 48) = 97.
    
    # Strassen 4x4x4: 
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    
    # 7 products of 2x2 matrices, each done via Strassen (rank 7).
    # Total rank = 7 * 7 = 49.
    
    # Let me build this carefully.
    
    def strassen_4x4x4():
        """Build Strassen 4x4x4 decomposition. Rank 49.
        Returns U (16x49), V (16x49), W (16x49)
        A indices: i*4+j for i,j in 0..3 (row-major)
        B indices: j*4+k for j,k in 0..3
        C indices: k*4+i for k,i in 0..3 (following W[k*n+i] convention)
        """
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        # 2x2 Strassen
        sU, sV, sW = strassen_2x2_2x2()
        
        # Top-level Strassen: 7 products M1..M7
        # Each Mi is a 2x2x2 multiplication
        # We need to map the 2x2 blocks to the 4x4 indices
        
        # A11 = A[0:2, 0:2] -> indices 0,1,4,5
        # A12 = A[0:2, 2:4] -> indices 2,3,6,7
        # A21 = A[2:4, 0:2] -> indices 8,9,12,13
        # A22 = A[2:4, 2:4] -> indices 10,11,14,15
        
        # B11 = B[0:2, 0:2] -> indices 0,4,8,12 (B is 4x4, row-major: j*4+k)
        # B12 = B[0:2, 2:4] -> indices 2,6,10,14
        # B21 = B[2:4, 0:2] -> indices 1,5,9,13
        # B22 = B[2:4, 2:4] -> indices 3,7,11,15
        
        # C11 = C[0:2, 0:2] -> indices k*4+i for k in 0:2, i in 0:2 -> 0,1,4,5
        # C12 = C[0:2, 2:4] -> indices k*4+i for k in 0:2, i in 2:4 -> 2,3,6,7
        # C21 = C[2:4, 0:2] -> indices k*4+i for k in 2:4, i in 0:2 -> 8,9,12,13
        # C22 = C[2:4, 2:4] -> indices k*4+i for k in 2:4, i in 2:4 -> 10,11,14,15
        
        # Strassen formulas:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # C11 = M1 - M4 + M5 + M6
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 + M7
        
        # For each Mi, we use the 2x2x2 Strassen (rank 7) to decompose the 2x2 product.
        # The 2x2 Strassen gives 7 rank-1 terms for a 2x2x2 multiplication.
        
        # Let's define the block index mappings:
        # A blocks: A11=[0,1,4,5], A12=[2,3,6,7], A21=[8,9,12,13], A22=[10,11,14,15]
        # B blocks: B11=[0,4,8,12], B12=[2,6,10,14], B21=[1,5,9,13], B22=[3,7,11,15]
        # C blocks: C11=[0,1,4,5], C12=[2,3,6,7], C21=[8,9,12,13], C22=[10,11,14,15]
        
        # For each of the 7 top-level products, and for each of the 7 sub-level Strassen terms,
        # we add a rank-1 term.
        
        # The 2x2 Strassen sU, sV, sW have indices 0..3 for the 2x2 matrices.
        # For a product like (A11+A22)*(B11+B22):
        # The "A" part has indices corresponding to A11 and A22 blocks.
        # sU[i] gives the coefficient for A[i] in the 2x2 product.
        # We need to map 2x2 indices to 4x4 block indices.
        
        # Let me define a helper:
        # For a product P1 * P2 where P1 is a linear combination of A-blocks and P2 of B-blocks,
        # and the result contributes to C-blocks with coefficients:
        
        # This is getting very complex. Let me try a different approach.
        
        # I'll just use schoolbook for 4x4x4 (rank 64) and schoolbook for 4x4x3 (rank 48).
        # Total = 112. Same as seed.
        
        # To actually improve, I need a working Strassen implementation.
        # Let me try to directly construct the 49-rank terms.
        
        # Each rank-1 term in the 4x4x4 Strassen corresponds to:
        # - A linear form in the 16 A-entries
        # - A linear form in the 16 B-entries  
        # - A linear form in the 16 C-entries
        
        # The 49 terms come from 7 top-level * 7 sub-level.
        
        # Let me define the mappings explicitly.
        
        # 2x2 Strassen terms (sU, sV, sW as above):
        # sU = [
        #   [1,0,0,0],  # A00
        #   [0,0,0,0],  # A01
        #   [0,0,0,0],  # A10
        #   [1,0,0,0],  # A11
        # ]
        # Wait, let me re-extract from strassen_2x2_2x2()
        
        # From the function above:
        # Term 0: U[0]=1, U[3]=1; V[0]=1, V[3]=1; W[0]=1, W[3]=1
        # Term 1: U[2]=1, U[3]=1; V[0]=1; W[2]=1, W[3]=1
        # Term 2: U[0]=1; V[1]=1, V[3]=-1; W[1]=1, W[3]=-1
        # Term 3: U[3]=1; V[2]=1, V[0]=-1; W[0]=-1, W[2]=1
        # Term 4: U[0]=1, U[1]=1; V[3]=1; W[0]=1, W[1]=1
        # Term 5: U[2]=1, U[0]=-1; V[0]=1, V[1]=1; W[0]=1, W[2]=1
        # Term 6: U[1]=1, U[3]=-1; V[2]=1, V[3]=1; W[3]=1
        
        # So sU (4x7):
        # Row 0 (A00): [1, 0, 1, 0, 1, -1, 0]
        # Row 1 (A01): [0, 0, 0, 0, 1, 0, 1]
        # Row 2 (A10): [0, 1, 0, 0, 0, 1, 0]
        # Row 3 (A11): [1, 1, 0, 1, 0, 0, -1]
        
        # sV (4x7):
        # Row 0 (B00): [1, 1, 0, -1, 0, 1, 0]
        # Row 1 (B01): [0, 0, 1, 0, 0, 1, 0]
        # Row 2 (B10): [0, 0, 0, 1, 0, 0, 1]
        # Row 3 (B11): [1, 0, -1, 0, 1, 0, 1]
        
        # sW (4x7):
        # Row 0 (C00): [1, 0, 0, -1, 1, 1, 0]
        # Row 1 (C01): [0, 0, 1, 0, 1, 0, 0]
        # Row 2 (C10): [0, 1, 0, 1, 0, 1, 0]
        # Row 3 (C11): [1, 1, -1, 0, 0, 0, 1]
        
        # Now for the top-level Strassen:
        # M1 = (A11+A22)*(B11+B22)
        # The "A" for M1 is A11+A22, which in 2x2 block space is:
        # A_block = A11 + A22
        # In terms of 2x2 indices: the "2x2 matrix" for the A-side has entries:
        #   (0,0) = sum of A11[0,0] and A22[0,0]
        #   (0,1) = sum of A11[0,1] and A22[0,1]
        #   (1,0) = sum of A11[1,0] and A22[1,0]
        #   (1,1) = sum of A11[1,1] and A22[1,1]
        
        # So for the A-side of M1, the 2x2 index (r,c) maps to:
        #   A11[r,c] + A22[r,c]
        # In 4x4 indices: A11[r,c] is at index r*4+c (for r,c in 0,1)
        # A22[r,c] is at index (r+2)*4+(c+2)
        
        # Similarly for B-side and C-side.
        
        # Let me define the mapping for each top-level product.
        
        # Product definitions (A_blocks, B_blocks, C_contribution):
        # M1: A = A11+A22, B = B11+B22, C: C11 += 1, C22 += 1
        # M2: A = A21+A22, B = B11,       C: C21 += 1, C22 += 1
        # M3: A = A11,       B = B12-B22, C: C12 += 1, C22 -= 1
        # M4: A = A22,       B = B21-B11, C: C11 -= 1, C21 += 1
        # M5: A = A11+A12, B = B22,       C: C11 += 1, C12 += 1
        # M6: A = A21-A11, B = B11+B12, C: C11 += 1, C21 += 1
        # M7: A = A12-A22, B = B21+B22, C: C22 += 1
        
        # For each product, the A-side is a linear combination of 2x2 blocks,
        # and similarly for B-side. The C-side contributes to specific C-blocks.
        
        # Let me define helper functions to map 2x2 indices to 4x4 indices.
        
        def a_block_indices(block):
            """Return list of 4 4x4 A-indices for a 2x2 block."""
            if block == 'A11':
                return [0, 1, 4, 5]
            elif block == 'A12':
                return [2, 3, 6, 7]
            elif block == 'A21':
                return [8, 9, 12, 13]
            elif block == 'A22':
                return [10, 11, 14, 15]
        
        def b_block_indices(block):
            """Return list of 4 4x4 B-indices for a 2x2 block."""
            if block == 'B11':
                return [0, 4, 8, 12]
            elif block == 'B12':
                return [2, 6, 10, 14]
            elif block == 'B21':
                return [1, 5, 9, 13]
            elif block == 'B22':
                return [3, 7, 11, 15]
        
        def c_block_indices(block):
            """Return list of 4 4x4 C-indices for a 2x2 block."""
            if block == 'C11':
                return [0, 1, 4, 5]
            elif block == 'C12':
                return [2, 3, 6, 7]
            elif block == 'C21':
                return [8, 9, 12, 13]
            elif block == 'C22':
                return [10, 11, 14, 15]
        
        # For each top-level product, define:
        # a_coeffs: dict mapping 2x2 index (0..3) to list of (a_idx, coeff)
        # b_coeffs: dict mapping 2x2 index (0..3) to list of (b_idx, coeff)
        # c_coeffs: dict mapping 2x2 index (0..3) to list of (c_idx, coeff)
        
        products = [
            # M1
            {
                'a': {0: [('A11', 1), ('A22', 1)], 1: [('A11', 1), ('A22', 1)], 
                       2: [('A11', 1), ('A22', 1)], 3: [('A11', 1), ('A22', 1)]},
                'b': {0: [('B11', 1), ('B22', 1)], 1: [('B11', 1), ('B22', 1)], 
                       2: [('B11', 1), ('B22', 1)], 3: [('B11', 1), ('B22', 1)]},
                'c': {0: [('C11', 1)], 1: [], 2: [], 3: [('C22', 1)]}
            },
            # M2
            {
                'a': {0: [('A21', 1), ('A22', 1)], 1: [('A21', 1), ('A22', 1)], 
                       2: [('A21', 1), ('A22', 1)], 3: [('A21', 1), ('A22', 1)]},
                'b': {0: [('B11', 1)], 1: [], 2: [], 3: []},
                'c': {0: [], 1: [], 2: [('C21', 1)], 3: [('C22', 1)]}
            },
            # M3
            {
                'a': {0: [('A11', 1)], 1: [], 2: [], 3: []},
                'b': {0: [], 1: [('B12', 1), ('B22', -1)], 2: [], 3: []},
                'c': {0: [], 1: [('C12', 1)], 2: [], 3: [('C22', -1)]}
            },
            # M4
            {
                'a': {0: [], 1: [], 2: [], 3: [('A22', 1)]},
                'b': {0: [('B21', 1), ('B11', -1)], 1: [], 2: [], 3: []},
                'c': {0: [('C11', -1)], 1: [], 2: [('C21', 1)], 3: []}
            },
            # M5
            {
                'a': {0: [('A11', 1), ('A12', 1)], 1: [('A11', 1), ('A12', 1)], 
                       2: [('A11', 1), ('A12', 1)], 3: [('A11', 1), ('A12', 1)]},
                'b': {0: [], 1: [], 2: [], 3: [('B22', 1)]},
                'c': {0: [('C11', 1)], 1: [('C12', 1)], 2: [], 3: []}
            },
            # M6
            {
                'a': {0: [('A21', 1), ('A11', -1)], 1: [('A21', 1), ('A11', -1)], 
                       2: [('A21', 1), ('A11', -1)], 3: [('A21', 1), ('A11', -1)]},
                'b': {0: [('B11', 1), ('B12', 1)], 1: [('B11', 1), ('B12', 1)], 
                       2: [], 3: []},
                'c': {0: [('C11', 1)], 1: [], 2: [('C21', 1)], 3: []}
            },
            # M7
            {
                'a': {0: [('A12', 1), ('A22', -1)], 1: [('A12', 1), ('A22', -1)], 
                       2: [('A12', 1), ('A22', -1)], 3: [('A12', 1), ('A22', -1)]},
                'b': {0: [], 1: [], 2: [('B21', 1), ('B22', 1)], 3: [('B21', 1), ('B22', 1)]},
                'c': {0: [], 1: [], 2: [], 3: [('C22', 1)]}
            },
        ]
        
        # Now build the 49 rank-1 terms
        sU, sV, sW = strassen_2x2_2x2()
        
        for pi, prod in enumerate(products):
            for si in range(7):
                col = pi * 7 + si
                
                # A-side
                for a_2x2_idx in range(4):
                    coeff_a = sU[a_2x2_idx][si]
                    if coeff_a != 0:
                        for block, block_coeff in prod['a'][a_2x2_idx]:
                            for idx in a_block_indices(block):
                                U[idx][col] += coeff_a * block_coeff
                
                # B-side
                for b_2x2_idx in range(4):
                    coeff_b = sV[b_2x2_idx][si]
                    if coeff_b != 0:
                        for block, block_coeff in prod['b'][b_2x2_idx]:
                            for idx in b_block_indices(block):
                                V[idx][col] += coeff_b * block_coeff
                
                # C-side
                for c_2x2_idx in range(4):
                    coeff_c = sW[c_2x2_idx][si]
                    if coeff_c != 0:
                        for block, block_coeff in prod['c'][c_2x2_idx]:
                            for idx in c_block_indices(block):
                                W[idx][col] += coeff_c * block_coeff
        
        return U, V, W
    
    # Build 4x4x4 part (columns 0..3 of the 7-column output)
    U4, V4, W4 = strassen_4x4x4()
    
    # Build 4x4x3 part (columns 4..6 of the 7-column output) - schoolbook
    rank_4x4x3 = 4 * 4 * 3
    U3 = [[0]*rank_4x4x3 for _ in range(16)]
    V3 = [[0]*rank_4x4x3 for _ in range(16)]
    W3 = [[0]*rank_4x4x3 for _ in range(16)]
    
    col3 = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U3[i * 4 + j][col3] = 1
                V3[j * 4 + k][col3] = 1
                W3[k * 4 + i][col3] = 1
                col3 += 1
    
    # Combine: total rank = 49 + 48 = 97
    total_rank = 49 + 48
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(16)]
    W = [[0]*total_rank for _ in range(16)]
    
    for r in range(16):
        for c in range(49):
            U[r][c] = U4[r][c]
            V[r][c] = V4[r][c]
            W[r][c] = W4[r][c]
        for c in range(48):
            U[r][49 + c] = U3[r][c]
            V[r][49 + c] = V3[r][c]
            W[r][49 + c] = W3[r][c]
    
    if time.time() > deadline:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

