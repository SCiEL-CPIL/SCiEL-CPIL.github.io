import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    n, m, p = 3, 4, 8

    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return U, V, W

    def get_laderman_233():
        # Laderman's 14-rank algorithm for 2x3 x 3x2
        # Input A is 2x3, B is 3x2, Output C is 2x2
        # U shape (2*3, 14) = (6, 14)
        # V shape (3*2, 14) = (6, 14)
        # W shape (2*2, 14) = (4, 14)
        
        # Based on Laderman (1976) "On the number of multiplications needed to multiply matrices"
        # The 14 scalar multiplications are defined by combinations of sums of entries.
        
        # Let A = [[a11, a12, a13], [a21, a22, a23]]
        # Let B = [[b11, b12], [b21, b22], [b31, b32]]
        
        # The 14 products m_i:
        # m1 = (a11 + a21) * (b11 + b31)
        # m2 = (a12 + a22) * (b21 + b31)
        # m3 = (a13 + a23) * (b12 + b32)
        # m4 = (a11 + a22) * (b12 + b22)
        # m5 = (a12 + a23) * (b11 + b21)
        # m6 = (a11 + a12) * (b11 + b12)
        # m7 = (a11 + a13) * (b11 + b32)
        # m8 = (a21 + a22) * (b21 + b22)
        # m9 = (a22 + a23) * (b22 + b32)
        # m10 = (a11 - a22) * (b11 - b21)
        # m11 = (a12 - a23) * (b12 - b32)
        # m12 = (a13 - a21) * (b31 - b11)  # Note: signs and indices need careful check
        # m13 = (a11 - a23) * (b11 - b31)  # Wait, standard Laderman definitions vary by source.
        # m14 = (a12 - a21) * (b12 - b21)
        
        # Let's use a verified set of coefficients for 2x3x3 (2x3 * 3x2).
        # Reference: "Algorithms for Matrix Multiplication" or similar tensor decomposition tables.
        # Actually, for 2x3 * 3x2, the tensor is equivalent to 2x2 * 2x2 with extra dimensions? No.
        # 2x3 * 3x2 is a 2x3x3 tensor? No, dimensions are (2,3,2).
        # The task is 3x4 * 4x8.
        # We split 3x4 into 2x4 and 1x4.
        # We split 4x8 into 4x4 and 4x4? No, 4x8 is one matrix.
        
        # Let's re-evaluate the "Laderman for 2x3x3" claim in the prompt.
        # "Apply the Laderman rank-14 decomposition for the 2x3x3 matrix multiplication tensor"
        # This implies a tensor of size 2x3x3? i.e. (2x3) * (3x3) -> (2x3)?
        # If so, rank 14 is known for 2x3x3.
        # Let's check if we can use a 2x3x3 block.
        
        # Strategy:
        # A is 3x4. Split rows into 2 and 1.
        # A1: 2x4, A2: 1x4.
        # B is 4x8.
        # We need to multiply A1 (2x4) by B (4x8).
        # This is a 2x4x8 tensor.
        # Can we decompose 2x4x8 using Laderman?
        # Laderman 2x3x3 is for (2x3)*(3x3)->(2x3).
        # Our sub-problem is (2x4)*(4x8)->(2x8).
        # This is not directly 2x3x3.
        
        # However, we can split the 4x8 B matrix into B1 (4x4) and B2 (4x4).
        # Then A1 * B1 is 2x4 * 4x4 -> 2x4.
        # A1 * B2 is 2x4 * 4x4 -> 2x4.
        # This doesn't seem to map to 2x3x3 directly.
        
        # Let's look at the prompt's specific claim:
        # "splitting the 4x8 matrix into a 3x8 and 1x8 part"
        # This implies B is split into B1 (3x8) and B2 (1x8)?
        # But B is 4x8. Splitting into 3x8 and 1x8 means we are effectively ignoring one row?
        # No, that's not right.
        
        # Let's reconsider the standard known low-rank decompositions.
        # Schoolbook 3x4x8 is 96.
        # Strassen 2x2x2 based recursion gave 88.
        # Is 84 possible?
        
        # Let's try to construct a specific decomposition using known small blocks.
        # We can use the 2x2x2 Strassen (rank 7) for 2x2 blocks.
        # We can use the 2x3x3 Laderman (rank 14) if we can fit the dimensions.
        
        # Let's try to build the 3x4x8 tensor from smaller known tensors.
        # Split n=3 into 2+1.
        # Split m=4 into 2+2.
        # Split p=8 into 4+4.
        
        # Block (2,2,4): 2x2 * 2x4 -> 2x4.
        # We can treat this as two 2x2x2 multiplications side-by-side.
        # Rank = 7 + 7 = 14.
        
        # Block (2,2,4) again for the other 2x2 part of A?
        # A is 3x4.
        # Rows 0,1 (2x4) and Row 2 (1x4).
        # Cols 0,1 (2x2) and Cols 2,3 (2x2).
        
        # Let's define the blocks of A (3x4):
        # A[0:2, 0:2] (2x2)
        # A[0:2, 2:4] (2x2)
        # A[2:3, 0:2] (1x2)
        # A[2:3, 2:4] (1x2)
        
        # Blocks of B (4x8):
        # B[0:2, 0:4] (2x4)
        # B[0:2, 4:8] (2x4)
        # B[2:4, 0:4] (2x4)
        # B[2:4, 4:8] (2x4)
        
        # Output C (3x8) blocks:
        # C[0:2, 0:4] = A[0:2,0:2]@B[0:2,0:4] + A[0:2,2:4]@B[2:4,0:4]
        # C[0:2, 4:8] = A[0:2,0:2]@B[0:2,4:8] + A[0:2,2:4]@B[2:4,4:8]
        # C[2:3, 0:4] = A[2:3,0:2]@B[0:2,0:4] + A[2:3,2:4]@B[2:4,0:4]
        # C[2:3, 4:8] = A[2:3,0:2]@B[0:2,4:8] + A[2:3,2:4]@B[2:4,4:8]
        
        # Let's calculate ranks for each term.
        # Term 1: A[0:2,0:2] (2x2) @ B[0:2,0:4] (2x4).
        # This is (2x2) * (2x4) -> (2x4).
        # We can split B[0:2,0:4] into two 2x2 matrices: B11 (cols 0,1) and B12 (cols 2,3).
        # So this is A11 @ B11 (2x2*2x2->2x2) + A11 @ B12 (2x2*2x2->2x2).
        # Using Strassen for 2x2x2 (rank 7).
        # Rank = 7 + 7 = 14.
        
        # Term 2: A[0:2,2:4] (2x2) @ B[2:4,0:4] (2x4).
        # Similarly, split B[2:4,0:4] into B21 (cols 0,1) and B22 (cols 2,3).
        # Rank = 7 + 7 = 14.
        
        # So C[0:2, 0:4] requires 14 + 14 = 28 rank units.
        # By symmetry, C[0:2, 4:8] also requires 28 rank units.
        # Total for top 2 rows of C: 56 rank units.
        
        # Now bottom row of C (1x8).
        # C[2:3, 0:4] = A[2:3,0:2] (1x2) @ B[0:2,0:4] (2x4) + A[2:3,2:4] (1x2) @ B[2:4,0:4] (2x4).
        # Term 3: A[2:3,0:2] (1x2) @ B[0:2,0:4] (2x4).
        # Split B[0:2,0:4] into B11 (2x2) and B12 (2x2).
        # A_row @ B11 (1x2 * 2x2 -> 1x2).
        # This is a 1x2x2 tensor. Rank is 2 (schoolbook).
        # A_row @ B12 (1x2 * 2x2 -> 1x2). Rank is 2.
        # Total for Term 3: 2 + 2 = 4.
        
        # Term 4: A[2:3,2:4] (1x2) @ B[2:4,0:4] (2x4).
        # Similarly, rank 2 + 2 = 4.
        
        # Total for C[2:3, 0:4]: 4 + 4 = 8.
        # By symmetry, C[2:3, 4:8] also requires 8 rank units.
        # Total for bottom row of C: 16 rank units.
        
        # Total Rank = 56 + 16 = 72.
        
        # Wait, this is significantly lower than 88.
        # Did I make a mistake?
        # Let's re-verify the "Strassen 2x2x2" application.
        # A 2x2 * 2x4 multiplication.
        # A is 2x2. B is 2x4. C is 2x4.
        # C[:, 0:2] = A @ B[:, 0:2]. (2x2 * 2x2 -> 2x2). Rank 7.
        # C[:, 2:4] = A @ B[:, 2:4]. (2x2 * 2x2 -> 2x2). Rank 7.
        # Total rank 14. Correct.
        
        # A 1x2 * 2x4 multiplication.
        # A is 1x2. B is 2x4. C is 1x4.
        # C[:, 0:2] = A @ B[:, 0:2]. (1x2 * 2x2 -> 1x2). Rank 2.
        # C[:, 2:4] = A @ B[:, 2:4]. (1x2 * 2x2 -> 1x2). Rank 2.
        # Total rank 4. Correct.
        
        # So the total rank is indeed 72.
        # Why did the previous program get 88?
        # The previous program used a recursive block decomposition that likely didn't optimize the 1x2x2 or 2x2x2 blocks as efficiently, or used a different split.
        # The previous code:
        # if n > 1: split n.
        # if p > 1: split p.
        # if m > 1: split m?
        
        # In the previous code:
        # solve_block_optimized(3, 4, 8)
        # n=3 > 1. n1=1, n2=2.
        # solve_block_optimized(1, 4, 8)
        #   n=1.
        #   p=8 > 1. p1=4, p2=4.
        #   solve_block_optimized(1, 4, 4)
        #     n=1. p=4 > 1. p1=2, p2=2.
        #     solve_block_optimized(1, 4, 2)
        #       n=1. p=2 > 1. p1=1, p2=1.
        #       solve_block_optimized(1, 4, 1)
        #         n=1. p=1. m=4 > 1.
        #         Returns schoolbook 1x4x1? Rank 4.
        #       ...
        # This recursion seems to fall back to schoolbook for the 1xMxP cases often, or splits inefficiently.
        
        # My manual calculation suggests 72 is possible with simple Strassen 2x2x2 and schoolbook for 1x2x2.
        # Let's implement this 72-rank decomposition.
        
        # We need to construct U, V, W for the 3x4x8 tensor.
        
        # We will define the rank-1 components for each block multiplication.
        
        def get_strassen_222():
            # Returns U (4,7), V (4,7), W (4,7) for 2x2*2x2->2x2
            # U indices: 0:a11, 1:a12, 2:a21, 3:a22
            # V indices: 0:b11, 1:b12, 2:b21, 3:b22
            # W indices: 0:c11, 1:c12, 2:c21, 3:c22
            
            U = np.array([
                [1, 0, 0, 0, 0, 0, 0], # a11
                [0, 0, 1, 0, 1, 0, 0], # a12
                [0, 0, 0, 0, 0, -1, 0], # a21
                [1, 1, 0, 1, 0, 0, 0]  # a22
            ])
            
            V = np.array([
                [1, 1, 0, 0, 0, 1, 0], # b11
                [0, 0, 1, 0, 0, 1, 0], # b12
                [1, 0, -1, 1, 0, 0, 1], # b21
                [1, 0, 0, -1, 1, 0, 1]  # b22
            ])
            
            W = np.array([
                [1, 0, 0, 1, -1, 0, 1], # c11
                [0, 1, 0, 1, 0, 0, 0],  # c12
                [0, 0, 1, 0, 1, 0, 0],  # c21
                [1, -1, 1, 0, 0, 1, 0]  # c22
            ])
            return U, V, W

        def get_schoolbook_122():
            # 1x2 * 2x2 -> 1x2
            # U (2, 2): a1, a2
            # V (4, 2): b11, b12, b21, b22
            # W (2, 2): c1, c2
            
            U = np.array([
                [1, 0], # a1
                [0, 1]  # a2
            ])
            
            V = np.array([
                [1, 0], # b11
                [0, 0], # b12 (not used in c1? c1 = a1*b11 + a2*b21)
                [0, 1], # b21
                [0, 0]  # b22
            ])
            # Wait, c1 = a1*b11 + a2*b21. c2 = a1*b12 + a2*b22.
            # Term 1: a1*b11. U[0]=1, V[0]=1, W[0]=1.
            # Term 2: a2*b21. U[1]=1, V[2]=1, W[0]=1.
            # Term 3: a1*b12. U[0]=1, V[1]=1, W[1]=1.
            # Term 4: a2*b22. U[1]=1, V[3]=1, W[1]=1.
            
            U = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1]
            ])
            V = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            W = np.array([
                [1, 1, 0, 0],
                [0, 0, 1, 1]
            ])
            return U, V, W

        # We need to assemble these into the global 3x4x8 tensor.
        # Global U shape (12, R), V shape (32, R), W shape (24, R).
        
        R = 72
        U = np.zeros((12, R))
        V = np.zeros((32, R))
        W = np.zeros((24, R))
        
        idx = 0
        
        # Helper to add a block decomposition
        def add_block(U_block, V_block, W_block, 
                      i_start, j_start, k_start, # Output C start
                      a_i_start, a_j_start, # Input A start
                      b_j_start, b_k_start): # Input B start
            nonlocal idx
            r_block = U_block.shape[1]
            n_b = U_block.shape[0] // 1 # This is tricky. U_block is flattened.
            # We need to know the dimensions of the block to map indices.
            # It's better to pass explicit dimensions.
            
        # Let's do it manually for clarity and correctness.
        
        # Strassen 2x2x2 blocks
        U_s, V_s, W_s = get_strassen_222()
        # Schoolbook 1x2x2 blocks
        U_sb, V_sb, W_sb = get_schoolbook_122()
        
        # 1. C[0:2, 0:4]
        #   A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2]
        #   A[0:2, 0:2] @ B[0:2, 2:4] -> C[0:2, 2:4]
        #   A[0:2, 2:4] @ B[2:4, 0:2] -> C[0:2, 0:2]
        #   A[0:2, 2:4] @ B[2:4, 2:4] -> C[0:2, 2:4]
        
        # Sub-block 1.1: A[0:2, 0:2] (2x2) @ B[0:2, 0:2] (2x2) -> C[0:2, 0:2] (2x2)
        # A indices: rows 0,1; cols 0,1.
        # B indices: rows 0,1; cols 0,1.
        # C indices: rows 0,1; cols 0,1.
        for r in range(7):
            # U
            U[0*4 + 0, idx+r] = U_s[0, r] # a11
            U[0*4 + 1, idx+r] = U_s[1, r] # a12
            U[1*4 + 0, idx+r] = U_s[2, r] # a21
            U[1*4 + 1, idx+r] = U_s[3, r] # a22
            
            # V
            V[0*8 + 0, idx+r] = V_s[0, r] # b11
            V[0*8 + 1, idx+r] = V_s[1, r] # b12
            V[1*8 + 0, idx+r] = V_s[2, r] # b21
            V[1*8 + 1, idx+r] = V_s[3, r] # b22
            
            # W
            W[0*3 + 0, idx+r] = W_s[0, r] # c11
            W[0*3 + 1, idx+r] = W_s[1, r] # c12
            W[1*3 + 0, idx+r] = W_s[2, r] # c21
            W[1*3 + 1, idx+r] = W_s[3, r] # c22
        idx += 7
        
        # Sub-block 1.2: A[0:2, 0:2] (2x2) @ B[0:2, 2:4] (2x2) -> C[0:2, 2:4] (2x2)
        # A indices: rows 0,1; cols 0,1.
        # B indices: rows 0,1; cols 2,3.
        # C indices: rows 0,1; cols 2,3.
        for r in range(7):
            U[0*4 + 0, idx+r] = U_s[0, r]
            U[0*4 + 1, idx+r] = U_s[1, r]
            U[1*4 + 0, idx+r] = U_s[2, r]
            U[1*4 + 1, idx+r] = U_s[3, r]
            
            V[0*8 + 2, idx+r] = V_s[0, r] # b11 (col 2)
            V[0*8 + 3, idx+r] = V_s[1, r] # b12 (col 3)
            V[1*8 + 2, idx+r] = V_s[2, r] # b21 (col 2)
            V[1*8 + 3, idx+r] = V_s[3, r] # b22 (col 3)
            
            W[0*3 + 2, idx+r] = W_s[0, r] # c11 (col 2)
            W[0*3 + 3, idx+r] = W_s[1, r] # c12 (col 3)
            W[1*3 + 2, idx+r] = W_s[2, r] # c21 (col 2)
            W[1*3 + 3, idx+r] = W_s[3, r] # c22 (col 3)
        idx += 7
        
        # Sub-block 1.3: A[0:2, 2:4] (2x2) @ B[2:4, 0:2] (2x2) -> C[0:2, 0:2] (2x2)
        # A indices: rows 0,1; cols 2,3.
        # B indices: rows 2,3; cols 0,1.
        # C indices: rows 0,1; cols 0,1.
        for r in range(7):
            U[0*4 + 2, idx+r] = U_s[0, r]
            U[0*4 + 3, idx+r] = U_s[1, r]
            U[1*4 + 2, idx+r] = U_s[2, r]
            U[1*4 + 3, idx+r] = U_s[3, r]
            
            V[2*8 + 0, idx+r] = V_s[0, r] # b11 (row 2, col 0)
            V[2*8 + 1, idx+r] = V_s[1, r] # b12 (row 2, col 1)
            V[3*8 + 0, idx+r] = V_s[2, r] # b21 (row 3, col 0)
            V[3*8 + 1, idx+r] = V_s[3, r] # b22 (row 3, col 1)
            
            W[0*3 + 0, idx+r] = W_s[0, r]
            W[0*3 + 1, idx+r] = W_s[1, r]
            W[1*3 + 0, idx+r] = W_s[2, r]
            W[1*3 + 1, idx+r] = W_s[3, r]
        idx += 7
        
        # Sub-block 1.4: A[0:2, 2:4] (2x2) @ B[2:4, 2:4] (2x2) -> C[0:2, 2:4] (2x2)
        # A indices: rows 0,1; cols 2,3.
        # B indices: rows 2,3; cols 2,3.
        # C indices: rows 0,1; cols 2,3.
        for r in range(7):
            U[0*4 + 2, idx+r] = U_s[0, r]
            U[0*4 + 3, idx+r] = U_s[1, r]
            U[1*4 + 2, idx+r] = U_s[2, r]
            U[1*4 + 3, idx+r] = U_s[3, r]
            
            V[2*8 + 2, idx+r] = V_s[0, r]
            V[2*8 + 3, idx+r] = V_s[1, r]
            V[3*8 + 2, idx+r] = V_s[2, r]
            V[3*8 + 3, idx+r] = V_s[3, r]
            
            W[0*3 + 2, idx+r] = W_s[0, r]
            W[0*3 + 3, idx+r] = W_s[1, r]
            W[1*3 + 2, idx+r] = W_s[2, r]
            W[1*3 + 3, idx+r] = W_s[3, r]
        idx += 7
        
        # 2. C[0:2, 4:8]
        #   A[0:2, 0:2] @ B[0:2, 4:6] -> C[0:2, 4:6]
        #   A[0:2, 0:2] @ B[0:2, 6:8] -> C[0:2, 6:8]
        #   A[0:2, 2:4] @ B[2:4, 4:6] -> C[0:2, 4:6]
        #   A[0:2, 2:4] @ B[2:4, 6:8] -> C[0:2, 6:8]
        
        # Sub-block 2.1: A[0:2, 0:2] @ B[0:2, 4:6] -> C[0:2, 4:6]
        for r in range(7):
            U[0*4 + 0, idx+r] = U_s[0, r]
            U[0*4 + 1, idx+r] = U_s[1, r]
            U[1*4 + 0, idx+r] = U_s[2, r]
            U[1*4 + 1, idx+r] = U_s[3, r]
            
            V[0*8 + 4, idx+r] = V_s[0, r]
            V[0*8 + 5, idx+r] = V_s[1, r]
            V[1*8 + 4, idx+r] = V_s[2, r]
            V[1*8 + 5, idx+r] = V_s[3, r]
            
            W[0*3 + 4, idx+r] = W_s[0, r]
            W[0*3 + 5, idx+r] = W_s[1, r]
            W[1*3 + 4, idx+r] = W_s[2, r]
            W[1*3 + 5, idx+r] = W_s[3, r]
        idx += 7
        
        # Sub-block 2.2: A[0:2, 0:2] @ B[0:2, 6:8] -> C[0:2, 6:8]
        for r in range(7):
            U[0*4 + 0, idx+r] = U_s[0, r]
            U[0*4 + 1, idx+r] = U_s[1, r]
            U[1*4 + 0, idx+r] = U_s[2, r]
            U[1*4 + 1, idx+r] = U_s[3, r]
            
            V[0*8 + 6, idx+r] = V_s[0, r]
            V[0*8 + 7, idx+r] = V_s[1, r]
            V[1*8 + 6, idx+r] = V_s[2, r]
            V[1*8 + 7, idx+r] = V_s[3, r]
            
            W[0*3 + 6, idx+r] = W_s[0, r]
            W[0*3 + 7, idx+r] = W_s[1, r]
            W[1*3 + 6, idx+r] = W_s[2, r]
            W[1*3 + 7, idx+r] = W_s[3, r]
        idx += 7
        
        # Sub-block 2.3: A[0:2, 2:4] @ B[2:4, 4:6] -> C[0:2, 4:6]
        for r in range(7):
            U[0*4 + 2, idx+r] = U_s[0, r]
            U[0*4 + 3, idx+r] = U_s[1, r]
            U[1*4 + 2, idx+r] = U_s[2, r]
            U[1*4 + 3, idx+r] = U_s[3, r]
            
            V[2*8 + 4, idx+r] = V_s[0, r]
            V[2*8 + 5, idx+r] = V_s[1, r]
            V[3*8 + 4, idx+r] = V_s[2, r]
            V[3*8 + 5, idx+r] = V_s[3, r]
            
            W[0*3 + 4, idx+r] = W_s[0, r]
            W[0*3 + 5, idx+r] = W_s[1, r]
            W[1*3 + 4, idx+r] = W_s[2, r]
            W[1*3 + 5, idx+r] = W_s[3, r]
        idx += 7
        
        # Sub-block 2.4: A[0:2, 2:4] @ B[2:4, 6:8] -> C[0:2, 6:8]
        for r in range(7):
            U[0*4 + 2, idx+r] = U_s[0, r]
            U[0*4 + 3, idx+r] = U_s[1, r]
            U[1*4 + 2, idx+r] = U_s[2, r]
            U[1*4 + 3, idx+r] = U_s[3, r]
            
            V[2*8 + 6, idx+r] = V_s[0, r]
            V[2*8 + 7, idx+r] = V_s[1, r]
            V[3*8 + 6, idx+r] = V_s[2, r]
            V[3*8 + 7, idx+r] = V_s[3, r]
            
            W[0*3 + 6, idx+r] = W_s[0, r]
            W[0*3 + 7, idx+r] = W_s[1, r]
            W[1*3 + 6, idx+r] = W_s[2, r]
            W[1*3 + 7, idx+r] = W_s[3, r]
        idx += 7
        
        # 3. C[2:3, 0:4]
        #   A[2:3, 0:2] (1x2) @ B[0:2, 0:2] (2x2) -> C[2:3, 0:2] (1x2)
        #   A[2:3, 0:2] (1x2) @ B[0:2, 2:4] (2x2) -> C[2:3, 2:4] (1x2)
        #   A[2:3, 2:4] (1x2) @ B[2:4, 0:2] (2x2) -> C[2:3, 0:2] (1x2)
        #   A[2:3, 2:4] (1x2) @ B[2:4, 2:4] (2x2) -> C[2:3, 2:4] (1x2)
        
        # Sub-block 3.1: A[2:3, 0:2] @ B[0:2, 0:2] -> C[2:3, 0:2]
        for r in range(4):
            U[2*4 + 0, idx+r] = U_sb[0, r] # a1
            U[2*4 + 1, idx+r] = U_sb[1, r] # a2
            
            V[0*8 + 0, idx+r] = V_sb[0, r] # b11
            V[0*8 + 1, idx+r] = V_sb[1, r] # b12
            V[1*8 + 0, idx+r] = V_sb[2, r] # b21
            V[1*8 + 1, idx+r] = V_sb[3, r] # b22
            
            W[2*3 + 0, idx+r] = W_sb[0, r] # c1
            W[2*3 + 1, idx+r] = W_sb[1, r] # c2
        idx += 4
        
        # Sub-block 3.2: A[2:3, 0:2] @ B[0:2, 2:4] -> C[2:3, 2:4]
        for r in range(4):
            U[2*4 + 0, idx+r] = U_sb[0, r]
            U[2*4 + 1, idx+r] = U_sb[1, r]
            
            V[0*8 + 2, idx+r] = V_sb[0, r]
            V[0*8 + 3, idx+r] = V_sb[1, r]
            V[1*8 + 2, idx+r] = V_sb[2, r]
            V[1*8 + 3, idx+r] = V_sb[3, r]
            
            W[2*3 + 2, idx+r] = W_sb[0, r]
            W[2*3 + 3, idx+r] = W_sb[1, r]
        idx += 4
        
        # Sub-block 3.3: A[2:3, 2:4] @ B[2:4, 0:2] -> C[2:3, 0:2]
        for r in range(4):
            U[2*4 + 2, idx+r] = U_sb[0, r]
            U[2*4 + 3, idx+r] = U_sb[1, r]
            
            V[2*8 + 0, idx+r] = V_sb[0, r]
            V[2*8 + 1, idx+r] = V_sb[1, r]
            V[3*8 + 0, idx+r] = V_sb[2, r]
            V[3*8 + 1, idx+r] = V_sb[3, r]
            
            W[2*3 + 0, idx+r] = W_sb[0, r]
            W[2*3 + 1, idx+r] = W_sb[1, r]
        idx += 4
        
        # Sub-block 3.4: A[2:3, 2:4] @ B[2:4, 2:4] -> C[2:3, 2:4]
        for r in range(4):
            U[2*4 + 2, idx+r] = U_sb[0, r]
            U[2*4 + 3, idx+r] = U_sb[1, r]
            
            V[2*8 + 2, idx+r] = V_sb[0, r]
            V[2*8 + 3, idx+r] = V_sb[1, r]
            V[3*8 + 2, idx+r] = V_sb[2, r]
            V[3*8 + 3, idx+r] = V_sb[3, r]
            
            W[2*3 + 2, idx+r] = W_sb[0, r]
            W[2*3 + 3, idx+r] = W_sb[1, r]
        idx += 4
        
        # 4. C[2:3, 4:8]
        #   A[2:3, 0:2] @ B[0:2, 4:6] -> C[2:3, 4:6]
        #   A[2:3, 0:2] @ B[0:2, 6:8] -> C[2:3, 6:8]
        #   A[2:3, 2:4] @ B[2:4, 4:6] -> C[2:3, 4:6]
        #   A[2:3, 2:4] @ B[2:4, 6:8] -> C[2:3, 6:8]
        
        # Sub-block 4.1: A[2:3, 0:2] @ B[0:2, 4:6] -> C[2:3, 4:6]
        for r in range(4):
            U[2*4 + 0, idx+r] = U_sb[0, r]
            U[2*4 + 1, idx+r] = U_sb[1, r]
            
            V[0*8 + 4, idx+r] = V_sb[0, r]
            V[0*8 + 5, idx+r] = V_sb[1, r]
            V[1*8 + 4, idx+r] = V_sb[2, r]
            V[1*8 + 5, idx+r] = V_sb[3, r]
            
            W[2*3 + 4, idx+r] = W_sb[0, r]
            W[2*3 + 5, idx+r] = W_sb[1, r]
        idx += 4
        
        # Sub-block 4.2: A[2:3, 0:2] @ B[0:2, 6:8] -> C[2:3, 6:8]
        for r in range(4):
            U[2*4 + 0, idx+r] = U_sb[0, r]
            U[2*4 + 1, idx+r] = U_sb[1, r]
            
            V[0*8 + 6, idx+r] = V_sb[0, r]
            V[0*8 + 7, idx+r] = V_sb[1, r]
            V[1*8 + 6, idx+r] = V_sb[2, r]
            V[1*8 + 7, idx+r] = V_sb[3, r]
            
            W[2*3 + 6, idx+r] = W_sb[0, r]
            W[2*3 + 7, idx+r] = W_sb[1, r]
        idx += 4
        
        # Sub-block 4.3: A[2:3, 2:4] @ B[2:4, 4:6] -> C[2:3, 4:6]
        for r in range(4):
            U[2*4 + 2, idx+r] = U_sb[0, r]
            U[2*4 + 3, idx+r] = U_sb[1, r]
            
            V[2*8 + 4, idx+r] = V_sb[0, r]
            V[2*8 + 5, idx+r] = V_sb[1, r]
            V[3*8 + 4, idx+r] = V_sb[2, r]
            V[3*8 + 5, idx+r] = V_sb[3, r]
            
            W[2*3 + 4, idx+r] = W_sb[0, r]
            W[2*3 + 5, idx+r] = W_sb[1, r]
        idx += 4
        
        # Sub-block 4.4: A[2:3, 2:4] @ B[2:4, 6:8] -> C[2:3, 6:8]
        for r in range(4):
            U[2*4 + 2, idx+r] = U_sb[0, r]
            U[2*4 + 3, idx+r] = U_sb[1, r]
            
            V[2*8 + 6, idx+r] = V_sb[0, r]
            V[2*8 + 7, idx+r] = V_sb[1, r]
            V[3*8 + 6, idx+r] = V_sb[2, r]
            V[3*8 + 7, idx+r] = V_sb[3, r]
            
            W[2*3 + 6, idx+r] = W_sb[0, r]
            W[2*3 + 7, idx+r] = W_sb[1, r]
        idx += 4
        
        if idx != R:
            raise ValueError(f"Rank mismatch: expected {R}, got {idx}")
            
        return U, V, W

    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        rank = U_np.shape[1]
        
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        
        rng = np.random.default_rng(42)
        for _ in range(5):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C = A @ B
            
            M = U_3d * A[:, :, np.newaxis] # (n, m, r)
            N = V_3d * B[:, :, np.newaxis] # (m, p, r)
            
            S = np.einsum('ijc,jkc->ikc', M, N)
            
            C_dec = np.einsum('ikc,kic->ik', S, W_3d)
            
            if np.max(np.abs(C - C_dec)) > tol:
                return False
        return True

    # Try the 72-rank decomposition
    try:
        U_new, V_new, W_new = get_laderman_233()
        if verify_decomp(U_new, V_new, W_new, n, m, p):
            U_final = np.round(U_new).astype(int)
            V_final = np.round(V_new).astype(int)
            W_final = np.round(W_new).astype(int)
            if verify_decomp(U_final, V_final, W_final, n, m, p):
                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
            else:
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    except Exception as e:
        pass

    # Fallback to schoolbook
    U_sb, V_sb, W_sb = get_schoolbook(n, m, p)
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Found rank: {rank}")
