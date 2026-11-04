"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strategy: Use rank-49 Strassen for M_{4,4,4} + 48 schoolbook for extra 3 columns
    # Total rank: 49 + 48 = 97
    
    def strassen_2x2x2():
        """Return rank-7 decomposition for M_{2,2,2}."""
        # U: (4,7), V: (4,7), W: (4,7)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 terms:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # a=U[0], b=U[1], c=U[2], d=U[3]
        # e=V[0], f=V[1], g=V[2], h=V[3]
        # C00=W[0], C01=W[1], C10=W[2], C11=W[3]
        
        # M1: (a+d)(e+h)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # M2: (c+d)e
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1; W[3,1] = -1
        
        # M3: a(f-h)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1
        
        # M4: d(g-e)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1
        
        # M5: (a+b)h
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        
        # M6: (c-a)(e+f)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # M7: (b-d)(g+h)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1
        
        return U, V, W
    
    def strassen_4x4x4():
        """Return rank-49 decomposition for M_{4,4,4} using recursive Strassen."""
        # Split 4x4 into 2x2 blocks
        # A = [[A00, A01], [A10, A11]], each block 2x2
        # B = [[B00, B01], [B10, B11]], each block 2x2
        # C = [[C00, C01], [C10, C11]], each block 2x2
        
        # Strassen at block level:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mj is a 2x2 matrix product, computed with rank-7 Strassen.
        # Total rank = 7 * 7 = 49.
        
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        # U index: i*4+j for 4x4 matrix
        # V index: j*4+k for 4x4 matrix
        # W index: k*4+i for 4x4 matrix
        
        # Block A00: rows 0-1, cols 0-1 -> U indices 0,1,4,5
        # Block A01: rows 0-1, cols 2-3 -> U indices 2,3,6,7
        # Block A10: rows 2-3, cols 0-1 -> U indices 8,9,12,13
        # Block A11: rows 2-3, cols 2-3 -> U indices 10,11,14,15
        
        # Block B00: rows 0-1, cols 0-1 -> V indices 0,1,4,5
        # Block B01: rows 0-1, cols 2-3 -> V indices 2,3,6,7
        # Block B10: rows 2-3, cols 0-1 -> V indices 8,9,12,13
        # Block B11: rows 2-3, cols 2-3 -> V indices 10,11,14,15
        
        # Block C00: rows 0-1, cols 0-1 -> W indices 0,1,4,5
        # Block C01: rows 0-1, cols 2-3 -> W indices 2,3,6,7
        # Block C10: rows 2-3, cols 0-1 -> W indices 8,9,12,13
        # Block C11: rows 2-3, cols 2-3 -> W indices 10,11,14,15
        
        U2, V2, W2 = strassen_2x2x2()
        
        # For each Strassen term Mj (j=0..6), we need to compute a 2x2 product.
        # The 2x2 product uses rank-7 Strassen, so for each Mj, we add 7 rank-1 terms.
        
        # Define the block indices
        # A blocks: [A00, A01, A10, A11] -> U row offsets [0, 2, 8, 10]
        # B blocks: [B00, B01, B10, B11] -> V row offsets [0, 2, 8, 10]
        # C blocks: [C00, C01, C10, C11] -> W row offsets [0, 2, 8, 10]
        
        # For a 2x2 block at position (row_block, col_block) in a 4x4 matrix:
        # U rows: row_block*4 + col_block*2 + [0,1,2,3]
        # Actually: U[i*4+j] where i is row, j is col
        # Block (rb, cb): rows rb*2 to rb*2+1, cols cb*2 to cb*2+1
        # U indices: (rb*2)*4 + cb*2 + [0,1,2,3] = rb*8 + cb*2 + [0,1,2,3]
        
        def u_block_indices(rb, cb):
            """Return U row indices for block (rb, cb) in 4x4."""
            base = rb * 8 + cb * 2
            return [base, base+1, base+2, base+3]
        
        def v_block_indices(rb, cb):
            """Return V row indices for block (rb, cb) in 4x4."""
            # V[j*4+k] where j is row of B, k is col of B
            base = rb * 8 + cb * 2
            return [base, base+1, base+2, base+3]
        
        def w_block_indices(rb, cb):
            """Return W row indices for block (rb, cb) in 4x4."""
            # W[k*4+i] where k is row of C, i is col of C
            base = rb * 8 + cb * 2
            return [base, base+1, base+2, base+3]
        
        # Strassen terms at block level:
        # M0: (A00+A11)(B00+B11)
        # M1: (A10+A11)B00
        # M2: A00(B10-B11)
        # M3: A11(B01-B00)
        # M4: (A00+A01)B11
        # M5: (A10-A00)(B00+B10)
        # M6: (A01-A11)(B01+B11)
        
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        # For each Mj, we compute a 2x2 product using Strassen (rank 7).
        # The 2x2 product has U, V, W of shape (4,7).
        # We need to map the 2x2 indices to 4x4 block indices.
        
        # For M0 = (A00+A11)(B00+B11):
        # The "A" part is A00+A11, which involves U rows from blocks A00 and A11.
        # The "B" part is B00+B11, which involves V rows from blocks B00 and B11.
        # The "C" part contributes to C00, C01, C10, C11.
        
        # This is getting complex. Let me use a different approach.
        
        # Instead, let me directly construct the rank-49 decomposition by
        # composing the Strassen decomposition at two levels.
        
        # Level 1: 7 Strassen terms for 4x4 (treating 2x2 blocks as "scalars")
        # Level 2: For each Strassen term, use rank-7 Strassen for 2x2 products
        
        # The composition: if T1 is rank r1 for M_{2,2,2} (block level) and
        # T2 is rank r2 for M_{2,2,2} (element level), then the composed
        # decomposition for M_{4,4,4} has rank r1*r2 = 49.
        
        # Let me define the block-level Strassen decomposition explicitly.
        
        # Block-level: treat each 2x2 block as a "scalar".
        # A_blocks: A00, A01, A10, A11 (4 "scalars")
        # B_blocks: B00, B01, B10, B11 (4 "scalars")
        # C_blocks: C00, C01, C10, C11 (4 "scalars")
        
        # Strassen at block level (same as 2x2 Strassen):
        # S0 = (A00+A11)(B00+B11)
        # S1 = (A10+A11)B00
        # S2 = A00(B10-B11)
        # S3 = A11(B01-B00)
        # S4 = (A00+A01)B11
        # S5 = (A10-A00)(B00+B10)
        # S6 = (A01-A11)(B01+B11)
        
        # C00 = S0 + S3 - S4 + S6
        # C01 = S2 + S4
        # C10 = S1 + S3
        # C11 = S0 - S1 + S2 + S5
        
        # Each Sj is a 2x2 matrix product.
        
        # Now, for each Sj, I need to express it as a rank-7 decomposition.
        # The 2x2 product Sj = Xj * Yj where Xj and Yj are 2x2 matrices
        # (linear combinations of A and B blocks).
        
        # For the tensor decomposition, each rank-1 term in the 2x2 Strassen
        # decomposition corresponds to a product of linear forms in A and B.
        # When we "inflate" each block to a 2x2 matrix, each linear form
        # becomes a sum of entries, and the rank-1 term becomes a sum of
        # rank-1 terms in the full 4x4 tensor.
        
        # This is the tensor product of the two Strassen decompositions.
        
        # Let me just hardcode the rank-49 decomposition.
        
        # Actually, let me use a simpler approach: just use schoolbook for M_{4,4,4}
        # which has rank 64, and add 48 for the extra 3 columns, total 112.
        # But that's not better than the baseline.
        
        # Let me try to implement the tensor product properly.
        
        # U4 shape: (16, 49)
        # V4 shape: (16, 49)
        # W4 shape: (16, 49)
        
        # For each of the 7 block-level Strassen terms (s=0..6), and each of
        # the 7 element-level Strassen terms (t=0..6), we have a rank-1 term
        # in the 4x4 decomposition at column s*7+t.
        
        # The block-level Strassen defines how A_blocks and B_blocks combine.
        # The element-level Strassen defines how entries within blocks combine.
        
        # For block-level term s, we have coefficients for each A_block and B_block.
        # For element-level term t, we have coefficients for each entry in the 2x2 blocks.
        
        # The combined coefficient for U[i*4+j, s*7+t] is:
        #   (block-level A coefficient for the block containing (i,j)) *
        #   (element-level U coefficient for the position of (i,j) within its block)
        
        # Similarly for V and W.
        
        # Let me define the block-level Strassen coefficients.
        
        # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
        #                0=B00, 1=B01, 2=B10, 3=B11
        #                0=C00, 1=C01, 2=C10, 3=C11
        
        # S0 = (A00+A11)(B00+B11): A coeffs [1,0,0,1], B coeffs [1,0,0,1]
        # S1 = (A10+A11)B00: A coeffs [0,0,1,1], B coeffs [1,0,0,0]
        # S2 = A00(B10-B11): A coeffs [1,0,0,0], B coeffs [0,0,1,-1]
        # S3 = A11(B01-B00): A coeffs [0,0,0,1], B coeffs [-1,1,0,0]
        # S4 = (A00+A01)B11: A coeffs [1,1,0,0], B coeffs [0,0,0,1]
        # S5 = (A10-A00)(B00+B10): A coeffs [-1,0,1,0], B coeffs [1,0,1,0]
        # S6 = (A01-A11)(B01+B11): A coeffs [0,1,0,-1], B coeffs [0,1,0,1]
        
        # C block contributions:
        # C00 = S0 + S3 - S4 + S6: S coeffs [1,0,0,1,-1,0,1]
        # C01 = S2 + S4: S coeffs [0,0,1,0,1,0,0]
        # C10 = S1 + S3: S coeffs [0,1,0,1,0,0,0]
        # C11 = S0 - S1 + S2 + S5: S coeffs [1,-1,1,0,0,1,0]
        
        block_A = np.array([
            [1, 0, 0, 1],  # S0
            [0, 0, 1, 1],  # S1
            [1, 0, 0, 0],  # S2
            [0, 0, 0, 1],  # S3
            [1, 1, 0, 0],  # S4
            [-1, 0, 1, 0], # S5
            [0, 1, 0, -1], # S6
        ])
        
        block_B = np.array([
            [1, 0, 0, 1],  # S0
            [1, 0, 0, 0],  # S1
            [0, 0, 1, -1], # S2
            [-1, 1, 0, 0], # S3
            [0, 0, 0, 1],  # S4
            [1, 0, 1, 0],  # S5
            [0, 1, 0, 1],  # S6
        ])
        
        block_C = np.array([
            [1, 0, 0, 1, -1, 0, 1],   # C00
            [0, 0, 1, 0, 1, 0, 0],   # C01
            [0, 1, 0, 1, 0, 0, 0],   # C10
            [1, -1, 1, 0, 0, 1, 0],  # C11
        ])
        
        # Element-level Strassen (2x2):
        # U2, V2, W2 from strassen_2x2x2()
        # U2[i*2+j, t] for i,j in {0,1}, t in {0..6}
        # V2[j*2+k, t]
        # W2[k*2+i, t]
        
        # Now, for the 4x4 decomposition:
        # U4[i*4+j, s*7+t] = block_A[s, block_of_(i,j)] * U2[(i%2)*2 + (j%2), t]
        # V4[j*4+k, s*7+t] = block_B[s, block_of_(j,k)] * V2[(j%2)*2 + (k%2), t]
        # W4[k*4+i, s*7+t] = block_C[block_of_(k,i), s] * W2[(k%2)*2 + (i%2), t]
        
        # Block of (i,j) in 4x4: (i//2, j//2) -> index (i//2)*2 + (j//2)
        # So block index = (i//2)*2 + (j//2)
        
        for i in range(4):
            for j in range(4):
                a_block = (i // 2) * 2 + (j // 2)
                u_pos = (i % 2) * 2 + (j % 2)
                for s in range(7):
                    for t in range(7):
                        col = s * 7 + t
                        U4[i * 4 + j, col] = block_A[s, a_block] * U2[u_pos, t]
        
        for j in range(4):
            for k in range(4):
                b_block = (j // 2) * 2 + (k // 2)
                v_pos = (j % 2) * 2 + (k % 2)
                for s in range(7):
                    for t in range(7):
                        col = s * 7 + t
                        V4[j * 4 + k, col] = block_B[s, b_block] * V2[v_pos, t]
        
        for k in range(4):
            for i in range(4):
                c_block = (k // 2) * 2 + (i // 2)
                w_pos = (k % 2) * 2 + (i % 2)
                for s in range(7):
                    for t in range(7):
                        col = s * 7 + t
                        W4[k * 4 + i, col] = block_C[c_block, s] * W2[w_pos, t]
        
        return U4, V4, W4
    
    # Build the full M_{4,4,7} decomposition
    # Part 1: M_{4,4,4} for columns 0-3, rank 49
    # Part 2: M_{4,4,3} for columns 4-6, rank 48 (schoolbook)
    
    U4, V4, W4 = strassen_4x4x4()
    
    total_rank = 49 + 48
    U = np.zeros((16, total_rank))
    V = np.zeros((28, total_rank))
    W = np.zeros((28, total_rank))
    
    # Copy the M_{4,4,4} part
    U[:, :49] = U4
    V[:, :49] = V4
    W[:, :49] = W4
    
    # Add schoolbook terms for columns 4,5,6
    col_offset = 49
    for k in range(3):  # k = 4,5,6 in full matrix
        k_full = k + 4
        for i in range(4):
            for j in range(4):
                U[i * 4 + j, col_offset] = 1
                V[j * 4 + k_full, col_offset] = 1
                W[k_full * 4 + i, col_offset] = 1
                col_offset += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

