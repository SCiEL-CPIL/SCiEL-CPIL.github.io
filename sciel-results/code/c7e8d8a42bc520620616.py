"""
Structural change: Instead of a pure schoolbook (rank=96) or a naive block decomposition,
we use a hybrid approach:
1. Recognize that the 4x4x6 multiplication can be viewed as two independent 4x4x2 multiplications
   (columns 0-3 of B and columns 4-5 of B), each of which has a tensor structure isomorphic to 
   a 4x4x2 matrix multiplication.
   
   However, a more direct structural improvement comes from using Strassen's algorithm 
   on the "square" part of the computation. The 4x4x6 matrix multiplication involves:
   C_{ik} = sum_j A_{ij} B_{jk} for i=0..3, j=0..3, k=0..5.
   
   We can decompose this into:
   - For k=0..3: C_{i, 0..3} = (A * B_{0..3})_{i, 0..3}  -> 4x4x4 multiplication
   - For k=4..5: C_{i, 4..5} = (A * B_{4..5})_{i, 4..5}  -> 4x4x2 multiplication
   
   For the 4x4x4 part, we can use Strassen's algorithm (rank 7 instead of 16).
   For the 4x4x2 part, the schoolbook rank is 4*4*2 = 32. But we can potentially do better.
   
   Actually, let's think about this differently. The tensor for 4x4x6 multiplication has
   dimension 4*4*6 = 96 for schoolbook. 
   
   A key insight: the 4x4x6 multiplication can be embedded in a 4x4x8 multiplication, but that's worse.
   
   Alternative structural approach: Use the fact that matrix multiplication is bilinear,
   and apply a recursive/block decomposition. But for small sizes like 4x4x6, the best known
   ranks are often achieved by specific constructions.
   
   Let me try a different structural approach: use a "slice" or "fiber" based decomposition.
   
   Actually, the most promising structural change for this specific problem is to use 
   Strassen's algorithm on the 4x4x4 sub-problem (which appears when we consider the first 4 
   columns of B) and keep the schoolbook for the remaining 2 columns.
   
   4x4x4 schoolbook: rank 64
   4x4x4 Strassen: rank 7 (for the 2x2x2 base case, scaled up)
   
   Wait, Strassen's algorithm for 2x2x2 has rank 7. For 4x4x4, we can apply Strassen 
   recursively: treat each 4x4 as a block of 2x2 blocks. This gives rank 7^2 = 49 for 
   the 4x4x4 part.
   
   For the 4x4x2 part (A is 4x4, B is 4x2, C is 4x2): schoolbook rank = 4*4*2 = 32.
   
   Total rank: 49 + 32 = 81, which is better than 96.
   
   Let me implement this.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll build the decomposition in two parts:
    # Part 1: 4x4x4 multiplication using Strassen (rank 49)
    # Part 2: 4x4x2 multiplication using schoolbook (rank 32)
    
    # Initialize with schoolbook as fallback
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Now let's try the hybrid Strassen approach
    # Part 1: 4x4x4 using Strassen
    # Strassen's algorithm for 2x2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # For 4x4x4, we apply Strassen recursively: each 4x4 matrix is a 2x2 block of 2x2 matrices.
    # This gives us a rank-7 decomposition at each level, so total rank = 7 * 7 = 49.
    
    # Let's build the Strassen decomposition for 4x4x4.
    # We'll represent the Strassen steps explicitly.
    
    def strassen_2x2_decomposition():
        """Return (U, V, W) for 2x2x2 Strassen, with indices:
        U indexed by (i*m+j) where m=2, so U[i*2+j, col]
        V indexed by (j*p+k) where p=2, so V[j*2+k, col]
        W indexed by (k*n+i) where n=2, so W[k*2+i, col]
        """
        rank = 7
        U = [[0] * rank for _ in range(4)]  # 2*2 = 4
        V = [[0] * rank for _ in range(4)]  # 2*2 = 4
        W = [[0] * rank for _ in range(4)]  # 2*2 = 4
        
        # P1 = (A00+A11)(B00+B11)
        # U: A00 (0,0), A11 (1,1)
        U[0][0] = 1  # A00
        U[3][0] = 1  # A11
        # V: B00 (0,0), B11 (1,1)
        V[0][0] = 1  # B00
        V[3][0] = 1  # B11
        # W: C00 (0,0), C11 (1,1)
        W[0][0] = 1  # C00
        W[3][0] = 1  # C11
        
        # P2 = (A10+A11)B00
        U[2][1] = 1  # A10
        U[3][1] = 1  # A11
        V[0][1] = 1  # B00
        W[2][1] = 1  # C10
        
        # P3 = A00(B10-B11)
        U[0][2] = 1  # A00
        V[2][2] = 1  # B10
        V[3][2] = -1 # B11
        W[1][2] = 1  # C01
        
        # P4 = A11(B01-B00)
        U[3][3] = 1  # A11
        V[1][3] = 1  # B01
        V[0][3] = -1 # B00
        W[0][3] = 1  # C00
        W[2][3] = 1  # C10
        
        # P5 = (A00+A01)B11
        U[0][4] = 1  # A00
        U[1][4] = 1  # A01
        V[3][4] = 1  # B11
        W[0][4] = -1 # C00
        W[1][4] = 1  # C01
        
        # P6 = (A10-A00)(B00+B10)
        U[2][5] = 1  # A10
        U[0][5] = -1 # A00
        V[0][5] = 1  # B00
        V[2][5] = 1  # B10
        W[3][5] = 1  # C11
        
        # P7 = (A01-A11)(B01+B11)
        U[1][6] = 1  # A01
        U[3][6] = -1 # A11
        V[1][6] = 1  # B01
        V[3][6] = 1  # B11
        W[0][6] = 1  # C00
        W[3][6] = 1  # C11
        
        return U, V, W
    
    def strassen_4x4_decomposition():
        """Build 4x4x4 Strassen decomposition by applying 2x2 Strassen recursively.
        Each 4x4 matrix is a 2x2 block of 2x2 matrices.
        """
        # The 4x4x4 multiplication can be seen as a 2x2 block multiplication 
        # where each "entry" is a 2x2 matrix multiplication.
        # We use Strassen at the block level, and within each block multiplication,
        # we can use either schoolbook (rank 4) or Strassen (rank 7).
        
        # To get rank 49, we use Strassen at both levels:
        # Level 1: 7 block multiplications of 2x2x2
        # Level 2: each 2x2x2 uses Strassen with rank 7
        # Total rank: 7 * 7 = 49
        
        # Let's build this explicitly.
        # A 4x4 matrix A is viewed as:
        # A = [[A00, A01],
        #      [A10, A11]]
        # where each Aij is a 2x2 matrix.
        
        # Similarly for B and C.
        
        # The Strassen formulas at the block level are the same as 2x2, but each 
        # "entry" is a 2x2 matrix, and the products are 2x2x2 matrix multiplications.
        
        # We'll enumerate all 49 terms.
        rank = 49
        U = [[0] * rank for _ in range(16)]  # 4*4 = 16
        V = [[0] * rank for _ in range(16)]  # 4*4 = 16
        W = [[0] * rank for _ in range(16)]  # 4*4 = 16
        
        # Helper: get local indices from global indices
        # For a 4x4 matrix with block size 2x2:
        # Global index (I, J) where I, J in 0..3
        # Block indices: (I//2, J//2)
        # Local indices within block: (I%2, J%2)
        
        # Strassen block formulas:
        # P1_block = (A00_block + A11_block) * (B00_block + B11_block)
        # P2_block = (A10_block + A11_block) * B00_block
        # P3_block = A00_block * (B10_block - B11_block)
        # P4_block = A11_block * (B01_block - B00_block)
        # P5_block = (A00_block + A01_block) * B11_block
        # P6_block = (A10_block - A00_block) * (B00_block + B10_block)
        # P7_block = (A01_block - A11_block) * (B01_block + B11_block)
        
        # For each block multiplication, we expand into 2x2x2 Strassen terms.
        
        # Let's define the 7 Strassen 2x2x2 decompositions explicitly.
        # Each term has coefficients for A (4 elements), B (4 elements), C (4 elements).
        
        # 2x2x2 Strassen terms (local indices 0..3 for 2x2 matrices):
        # Term 0: P1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        #   A: [1, 0, 0, 1], B: [1, 0, 0, 1], C: [1, 0, 0, 1]
        # Term 1: P2 = (A10+A11)B00 -> C10 += 1
        #   A: [0, 0, 1, 1], B: [1, 0, 0, 0], C: [0, 0, 1, 0]
        # Term 2: P3 = A00(B10-B11) -> C01 += 1
        #   A: [1, 0, 0, 0], B: [0, 0, 1, -1], C: [0, 1, 0, 0]
        # Term 3: P4 = A11(B01-B00) -> C00 += 1, C10 += 1
        #   A: [0, 0, 0, 1], B: [0, 1, -1, 0], C: [1, 0, 1, 0]
        # Term 4: P5 = (A00+A01)B11 -> C00 += -1, C01 += 1
        #   A: [1, 1, 0, 0], B: [0, 0, 0, 1], C: [-1, 1, 0, 0]
        # Term 5: P6 = (A10-A00)(B00+B10) -> C11 += 1
        #   A: [0, 0, 1, -1], B: [1, 0, 1, 0], C: [0, 0, 0, 1]
        # Term 6: P7 = (A01-A11)(B01+B11) -> C00 += 1, C11 += 1
        #   A: [0, 1, 0, -1], B: [0, 1, 0, 1], C: [1, 0, 0, 1]
        
        # Wait, I need to be more careful. Let me re-derive.
        # For 2x2 matrices:
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = A*B = [[c00, c01], [c10, c11]]
        
        # Strassen:
        # P1 = (a00+a11)(b00+b11)
        # P2 = (a10+a11)b00
        # P3 = a00(b10-b11)
        # P4 = a11(b01-b00)
        # P5 = (a00+a01)b11
        # P6 = (a10-a00)(b00+b10)
        # P7 = (a01-a11)(b01+b11)
        
        # c00 = P1 + P4 - P5 + P7
        # c01 = P3 + P5
        # c10 = P2 + P4
        # c11 = P1 - P2 + P3 + P6
        
        # So for each P_s, we have:
        # P1: A coeffs: a00=1, a11=1; B coeffs: b00=1, b11=1; C coeffs: c00=1, c11=1
        # P2: A coeffs: a10=1, a11=1; B coeffs: b00=1; C coeffs: c10=1
        # P3: A coeffs: a00=1; B coeffs: b10=1, b11=-1; C coeffs: c01=1
        # P4: A coeffs: a11=1; B coeffs: b01=1, b00=-1; C coeffs: c00=1, c10=1
        # P5: A coeffs: a00=1, a01=1; B coeffs: b11=1; C coeffs: c00=-1, c01=1
        # P6: A coeffs: a10=1, a00=-1; B coeffs: b00=1, b10=1; C coeffs: c11=1
        # P7: A coeffs: a01=1, a11=-1; B coeffs: b01=1, b11=1; C coeffs: c00=1, c11=1
        
        # Local index mapping for 2x2:
        # a00 -> idx 0, a01 -> idx 1, a10 -> idx 2, a11 -> idx 3
        
        # Let's define the 7 terms as (A_coeffs, B_coeffs, C_coeffs)
        strassen_2x2_terms = [
            # (A, B, C) as lists of length 4
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # P1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # P2
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),  # P3
            ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),  # P4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # P5
            ([0, 0, 1, -1], [1, 0, 1, 0], [0, 0, 0, 1]),  # P6
            ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 1]),  # P7
        ]
        
        # Now, for the 4x4x4 block Strassen, we have 7 block-level terms.
        # Each block-level term s has:
        # - A blocks involved: a subset of {00, 01, 10, 11}
        # - B blocks involved: a subset of {00, 01, 10, 11}
        # - C blocks involved: a subset of {00, 01, 10, 11}
        
        # Block-level Strassen:
        # P1_block: A00+A11, B00+B11, C00+=1, C11+=1
        # P2_block: A10+A11, B00, C10+=1
        # P3_block: A00, B10-B11, C01+=1
        # P4_block: A11, B01-B00, C00+=1, C10+=1
        # P5_block: A00+A01, B11, C00+=-1, C01+=1
        # P6_block: A10-A00, B00+B10, C11+=1
        # P7_block: A01-A11, B01+B11, C00+=1, C11+=1
        
        # For each block-level term s, and for each 2x2x2 Strassen term t (0..6),
        # we combine them to get a full 4x4x4 term.
        
        # Block-level term definitions:
        # (A_blocks, B_blocks, C_blocks) where each is a dict of block->coeff
        block_strassen = [
            # (A: {block: coeff}, B: {block: coeff}, C: {block: coeff})
            ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),  # P1
            ({'10': 1, '11': 1}, {'00': 1}, {'10': 1}),                      # P2
            ({'00': 1}, {'10': 1, '11': -1}, {'01': 1}),                     # P3
            ({'11': 1}, {'01': 1, '00': -1}, {'00': 1, '10': 1}),            # P4
            ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),            # P5
            ({'10': 1, '00': -1}, {'00': 1, '10': 1}, {'11': 1}),            # P6
            ({'01': 1, '11': -1}, {'01': 1, '11': 1}, {'00': 1, '11': 1}),   # P7
        ]
        
        col = 0
        for s in range(7):
            A_blocks, B_blocks, C_blocks = block_strassen[s]
            for t in range(7):
                A_coeffs_local, B_coeffs_local, C_coeffs_local = strassen_2x2_terms[t]
                
                # For each A block involved in term s, and each local index in that block:
                for A_block, A_coeff in A_blocks.items():
                    a_row, a_col = int(A_block[0]), int(A_block[1])
                    for li in range(2):
                        for lj in range(2):
                            if A_coeffs_local[li * 2 + lj] != 0:
                                # Global A index: (a_row * 2 + li) * 4 + (a_col * 2 + lj)
                                global_I = a_row * 2 + li
                                global_J = a_col * 2 + lj
                                U[global_I * 4 + global_J][col] += A_coeff * A_coeffs_local[li * 2 + lj]
                
                for B_block, B_coeff in B_blocks.items():
                    b_row, b_col = int(B_block[0]), int(B_block[1])
                    for lj in range(2):
                        for lk in range(2):
                            if B_coeffs_local[lj * 2 + lk] != 0:
                                # Global B index: (b_row * 2 + lj) * 4 + (b_col * 2 + lk)
                                global_J = b_row * 2 + lj
                                global_K = b_col * 2 + lk
                                V[global_J * 4 + global_K][col] += B_coeff * B_coeffs_local[lj * 2 + lk]
                
                for C_block, C_coeff in C_blocks.items():
                    c_row, c_col = int(C_block[0]), int(C_block[1])
                    for lk in range(2):
                        for li in range(2):
                            if C_coeffs_local[lk * 2 + li] != 0:
                                # Global C index: (c_col * 2 + lk) * 4 + (c_row * 2 + li)
                                # W is indexed by (k*n+i) = (k*4+i)
                                global_K = c_col * 2 + lk
                                global_I = c_row * 2 + li
                                W[global_K * 4 + global_I][col] += C_coeff * C_coeffs_local[lk * 2 + li]
                
                col += 1
        
        return U, V, W, col
    
    # Build the 4x4x4 part
    U44, V44, W44, rank44 = strassen_4x4_decomposition()
    
    # Build the 4x4x2 part (schoolbook)
    # A is 4x4, B is 4x2 (columns 4,5), C is 4x2 (columns 4,5)
    rank442 = 4 * 4 * 2
    U442 = [[0] * rank442 for _ in range(16)]
    V442 = [[0] * rank442 for _ in range(8)]  # 4*2 = 8
    W442 = [[0] * rank442 for _ in range(8)]  # 2*4 = 8
    
    col2 = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                # k is 0,1 corresponding to global columns 4,5
                U442[i * 4 + j][col2] = 1
                V442[j * 2 + k][col2] = 1
                W442[(4 + k) * 4 + i][col2] = 1  # global k is 4+k
                col2 += 1
    
    # Now combine into the full 4x4x6 decomposition
    total_rank = rank44 + rank442
    U_full = [[0] * total_rank for _ in range(16)]
    V_full = [[0] * total_rank for _ in range(24)]  # 4*6 = 24
    W_full = [[0] * total_rank for _ in range(24)]  # 6*4 = 24
    
    # Copy 4x4x4 part (columns 0-3 of B and C)
    for i in range(16):
        for c in range(rank44):
            U_full[i][c] = U44[i][c]
    
    for j in range(4):
        for k in range(4):
            for c in range(rank44):
                V_full[j * 6 + k][c] = V44[j * 4 + k][c]
    
    for k in range(4):
        for i in range(4):
            for c in range(rank44):
                W_full[k * 4 + i][c] = W44[k * 4 + i][c]
    
    # Copy 4x4x2 part (columns 4-5 of B and C)
    offset = rank44
    for i in range(16):
        for c in range(rank442):
            U_full[i][offset + c] = U442[i][c]
    
    for j in range(4):
        for k in range(2):
            for c in range(rank442):
                V_full[j * 6 + (4 + k)][offset + c] = V442[j * 2 + k][c]
    
    for k in range(2):
        for i in range(4):
            for c in range(rank442):
                W_full[(4 + k) * 4 + i][offset + c] = W442[k * 4 + i][c]
    
    # Return the hybrid decomposition
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
