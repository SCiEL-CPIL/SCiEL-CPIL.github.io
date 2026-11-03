"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit

    def verify(U, V, W, n, m, p):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    # Winograd-style block decomposition:
    # Embed the 3x5x6 multiplication into a 5x6x6 block structure.
    # The idea: partition rows/cols into blocks and apply low-rank decompositions
    # to the 3x3 sub-blocks using a rank-4 scheme (similar to Winograd/Fourier).
    #
    # Strategy: We can decompose the 3x5 * 5x6 product by splitting the middle
    # dimension (size 5) into blocks of size 2 and 1, or by splitting the
    # output dimensions.
    #
    # A more direct approach: use the fact that for a 3x3x3 tensor, we can
    # achieve rank 4 (the Winograd scheme for 3x3). For 3x5x6, we can:
    # - Split the 5 into 3+2 and the 6 into 3+3
    # - Apply rank-4 to the 3x3x3 blocks and Strassen to 2x2x2 blocks
    #
    # Let's implement a block decomposition:
    # Split n=3 into [3] (single block), m=5 into [3,2], p=6 into [3,3]
    # This gives blocks:
    #   (3,3,3): rank 4 (Winograd)
    #   (3,3,3): rank 4 (Winograd)
    #   (3,2,3): need to handle
    #   (3,2,3): need to handle
    #
    # Actually, let's use a cleaner approach: split m=5 into [2,3] and p=6 into [3,3]
    # Blocks:
    #   (3,2,3): use Strassen-like or direct
    #   (3,2,3): use Strassen-like or direct
    #   (3,3,3): rank 4
    #   (3,3,3): rank 4
    #
    # For (3,2,3): we can split further or use a direct rank-6 approach
    #
    # Let's try a different partition: n=3, m=5=[2,3], p=6=[2,4]
    # Blocks: (3,2,2), (3,2,4), (3,3,2), (3,3,4)
    #
    # Actually, the cleanest Winograd approach for 3x5x6:
    # Use the fact that 5 = 2+3 and 6 = 3+3
    # We'll construct the decomposition block by block.

    # Rank-4 decomposition for 3x3x3 (Winograd scheme)
    # This is based on the fact that the 3x3x3 multiplication tensor has rank 4
    # when using the Winograd Fourier transform approach.
    #
    # The 3x3x3 Winograd scheme:
    # For multiplying two 3x3 matrices, we can use 4 multiplications
    # by transforming to a different basis.
    #
    # The transformation matrices for 3x3 Winograd:
    # We use the DFT-like transform for size 3.
    # The 3x3 DFT matrix (with appropriate scaling) gives the Winograd scheme.

    def winograd_3x3():
        """Return rank-4 U, V, W for 3x3x3 matrix multiplication.
        U: (9, 4), V: (9, 4), W: (9, 4)
        Convention: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
        """
        # The 3x3 Winograd scheme uses the following approach:
        # We embed the 3x3 multiplication into a cyclic convolution structure.
        # The rank-4 decomposition corresponds to evaluating at 4 points.
        #
        # Using the Winograd algorithm for 3x3:
        # The transform matrices are based on the 3rd roots of unity and real/imag parts.
        #
        # Let me construct this explicitly.
        # The 3x3x3 tensor T[i,j,k] = 1 for all i,j,k in {0,1,2}
        # We want sum_r U[i*3+j,r] * V[j*3+k,r] * W[k*3+i,r] = 1
        #
        # One known rank-4 decomposition uses the following structure:
        # Based on the Winograd scheme, we can use:
        # r=0: U = [1,0,0, 0,0,0, 0,0,0]^T type patterns
        #
        # Let me use a direct construction.
        # The 3x3 matrix multiplication can be decomposed using the fact that:
        # C[i,k] = sum_j A[i,j] B[j,k]
        #
        # Using the Winograd approach with 4 terms:
        # We define the following rank-1 components:
        
        # Actually, let me verify: is the rank of 3x3x3 really 4?
        # The schoolbook rank is 27. The lower bound is max(9,9,9)=9.
        # The known rank for 3x3x3 is actually not 4 - that would be for a different problem.
        # The rank of the 3x3x3 matrix multiplication tensor is at least 9 (trivial)
        # and the best known is around 23 (from Strassen-style improvements).
        #
        # Wait - I was confusing this with something else. Let me reconsider.
        # The rank of the n x n x n matrix multiplication tensor is at least n^2
        # and at most n^3. For n=3, lower bound is 9, upper bound is 27.
        # The best known rank for 3x3x3 is 23 (from various improvements).
        #
        # So a rank-4 decomposition is NOT possible for 3x3x3.
        # The Winograd scheme for 3x3 convolution is different from matrix multiplication.
        #
        # Let me use a different approach: Strassen for 2x2x2 sub-blocks.

        # For now, return schoolbook for 3x3x3
        U = np.zeros((9, 27))
        V = np.zeros((9, 27))
        W = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j, col] = 1
                    V[j * 3 + k, col] = 1
                    W[k * 3 + i, col] = 1
                    col += 1
        return U, V, W

    # Strassen's rank-7 decomposition for 2x2x2
    def strassen_2x2():
        """Return rank-7 U, V, W for 2x2x2 matrix multiplication.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        # Strassen's algorithm: 7 multiplications for 2x2x2
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a11+a22)(b11+b22)
        # U: a11=1, a22=1 => U[0]=1, U[3]=1
        # V: b11=1, b22=1 => V[0]=1, V[3]=1
        # W: C11 += 1, C22 += 1 => W[0]=1, W[3]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (a21+a22)b11
        # U: a21=1, a22=1 => U[2]=1, U[3]=1
        # V: b11=1 => V[0]=1
        # W: C21 += 1, C22 -= 1 => W[2]=1, W[3]=-1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = a11(b12-b22)
        # U: a11=1 => U[0]=1
        # V: b12=1, b22=-1 => V[1]=1, V[3]=-1
        # W: C12 += 1, C22 += 1 => W[1]=1, W[3]=1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = a22(b21-b11)
        # U: a22=1 => U[3]=1
        # V: b21=1, b11=-1 => V[2]=1, V[0]=-1
        # W: C11 += 1, C21 += 1 => W[0]=1, W[2]=1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (a11+a12)b22
        # U: a11=1, a12=1 => U[0]=1, U[1]=1
        # V: b22=1 => V[3]=1
        # W: C11 -= 1, C12 += 1 => W[0]=-1, W[1]=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (a21-a11)(b11+b12)
        # U: a21=1, a11=-1 => U[2]=1, U[0]=-1
        # V: b11=1, b12=1 => V[0]=1, V[1]=1
        # W: C22 += 1 => W[3]=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (a12-a22)(b21+b22)
        # U: a12=1, a22=-1 => U[1]=1, U[3]=-1
        # V: b21=1, b22=1 => V[2]=1, V[3]=1
        # W: C11 += 1 => W[0]=1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Now construct the full 3x5x6 decomposition using blocks
    # Split: n=3 (single), m=5 = [2,3], p=6 = [3,3]
    # This gives 4 blocks:
    #   Block (0,0): n=3, m=2, p=3  -> rows 0-2, cols 0-1 of A, cols 0-2 of B
    #   Block (0,1): n=3, m=2, p=3  -> rows 0-2, cols 0-1 of A, cols 3-5 of B
    #   Block (1,0): n=3, m=3, p=3  -> rows 0-2, cols 2-4 of A, cols 0-2 of B
    #   Block (1,1): n=3, m=3, p=3  -> rows 0-2, cols 2-4 of A, cols 3-5 of B
    
    # For (3,2,3) blocks: schoolbook rank = 18
    # For (3,3,3) blocks: schoolbook rank = 27
    # Total schoolbook: 18+18+27+27 = 90 (same as before)
    
    # To improve, use Strassen on the 2x2x2 sub-blocks within the (3,2,3) blocks:
    # In a (3,2,3) block, we can split the n=3 into [2,1]:
    #   (2,2,3): split p=3 into [2,1]: (2,2,2) rank 7 + (2,2,1) rank 4
    #   (1,2,3): schoolbook rank 6
    # Total for (3,2,3): 7+4+6 = 17 (vs 18 schoolbook)
    
    # For (3,3,3) blocks: split n=3 into [2,1], m=3 into [2,1], p=3 into [2,1]
    # This gives 8 sub-blocks:
    #   (2,2,2): rank 7
    #   (2,2,1): rank 4
    #   (2,1,2): rank 4
    #   (2,1,1): rank 2
    #   (1,2,2): rank 4
    #   (1,2,1): rank 2
    #   (1,1,2): rank 2
    #   (1,1,1): rank 1
    # Total: 7+4+4+2+4+2+2+1 = 26 (vs 27 schoolbook)
    
    # Total with Strassen: 2*17 + 2*26 = 34 + 52 = 86
    
    # Let's implement this block decomposition:
    
    def construct_block_decomposition():
        """Construct the 3x5x6 decomposition using block-wise Strassen."""
        n, m, p = 3, 5, 6
        
        # We'll build U, V, W by concatenating block contributions
        # Block structure:
        # m split: [2, 3] (indices 0-1, 2-4)
        # p split: [3, 3] (indices 0-2, 3-5)
        
        # For each block, we further subdivide and apply Strassen where possible
        
        U_list = []
        V_list = []
        W_list = []
        
        def add_block(n_block, m_block, p_block, row_offset, col_offset_A, col_offset_B):
            """Add a block decomposition for an n_block x m_block x p_block sub-problem.
            row_offset: offset in the n dimension (for U rows and W rows)
            col_offset_A: offset in the m dimension (for U cols and V rows)
            col_offset_B: offset in the p dimension (for V cols and W cols)
            """
            # For small blocks, use Strassen or schoolbook
            
            if n_block == 2 and m_block == 2 and p_block == 2:
                # Use Strassen
                Ub, Vb, Wb = strassen_2x2()
                # Adjust indices
                U_adj = np.zeros((n * m, Ub.shape[1]))
                V_adj = np.zeros((m * p, Vb.shape[1]))
                W_adj = np.zeros((p * n, Wb.shape[1]))
                for i in range(n_block):
                    for j in range(m_block):
                        U_adj[(row_offset + i) * m + (col_offset_A + j), :] = Ub[i * m_block + j, :]
                for j in range(m_block):
                    for k in range(p_block):
                        V_adj[(col_offset_A + j) * p + (col_offset_B + k), :] = Vb[j * p_block + k, :]
                for k in range(p_block):
                    for i in range(n_block):
                        W_adj[(col_offset_B + k) * n + (row_offset + i), :] = Wb[k * n_block + i, :]
                U_list.append(U_adj)
                V_list.append(V_adj)
                W_list.append(W_adj)
                
            elif n_block == 2 and m_block == 2 and p_block == 1:
                # Schoolbook: rank 4
                rank = n_block * m_block * p_block
                Ub = np.zeros((n_block * m_block, rank))
                Vb = np.zeros((m_block * p_block, rank))
                Wb = np.zeros((p_block * n_block, rank))
                col = 0
                for i in range(n_block):
                    for j in range(m_block):
                        for k in range(p_block):
                            Ub[i * m_block + j, col] = 1
                            Vb[j * p_block + k, col] = 1
                            Wb[k * n_block + i, col] = 1
                            col += 1
                U_adj = np.zeros((n * m, rank))
                V_adj = np.zeros((m * p, rank))
                W_adj = np.zeros((p * n, rank))
                for i in range(n_block):
                    for j in range(m_block):
                        U_adj[(row_offset + i) * m + (col_offset_A + j), :] = Ub[i * m_block + j, :]
                for j in range(m_block):
                    for k in range(p_block):
                        V_adj[(col_offset_A + j) * p + (col_offset_B + k), :] = Vb[j * p_block + k, :]
                for k in range(p_block):
                    for i in range(n_block):
                        W_adj[(col_offset_B + k) * n + (row_offset + i), :] = Wb[k * n_block + i, :]
                U_list.append(U_adj)
                V_list.append(V_adj)
                W_list.append(W_adj)
                
            elif n_block == 1 and m_block == 2 and p_block == 2:
                # Schoolbook: rank 4
                rank = n_block * m_block * p_block
                Ub = np.zeros((n_block * m_block, rank))
                Vb = np.zeros((m_block * p_block, rank))
                Wb = np.zeros((p_block * n_block, rank))
                col = 0
                for i in range(n_block):
                    for j in range(m_block):
                        for k in range(p_block):
                            Ub[i * m_block + j, col] = 1
                            Vb[j * p_block + k, col] = 1
                            Wb[k * n_block + i, col] = 1
                            col += 1
                U_adj = np.zeros((n * m, rank))
                V_adj = np.zeros((m * p, rank))
                W_adj = np.zeros((p * n, rank))
                for i in range(n_block):
                    for j in range(m_block):
                        U_adj[(row_offset + i) * m + (col_offset_A + j), :] = Ub[i * m_block + j, :]
                for j in range(m_block):
                    for k in range(p_block):
                        V_adj[(col_offset_A + j) * p + (col_offset_B + k), :] = Vb[j * p_block + k, :]
                for k in range(p_block):
                    for i in range(n_block):
                        W_adj[(col_offset_B + k) * n + (row_offset + i), :] = Wb[k * n_block + i, :]
                U_list.append(U_adj)
                V_list.append(V_adj)
                W_list.append(W_adj)
                
            elif n_block == 2 and m_block == 1 and p_block == 2:
                # Schoolbook: rank 4
                rank = n_block * m_block * p_block
                Ub = np.zeros((n_block * m_block, rank))
                Vb = np.zeros((m_block * p_block, rank))
                Wb = np.zeros((p_block * n_block, rank))
                col = 0
                for i in range(n_block):
                    for j in range(m_block):
                        for k in range(p_block):
                            Ub[i * m_block + j, col] = 1
                            Vb[j * p_block + k, col] = 1
                            Wb[k * n_block + i, col] = 1
                            col += 1
                U_adj = np.zeros((n * m, rank))
                V_adj = np.zeros((m * p, rank))
                W_adj = np.zeros((p * n, rank))
                for i in range(n_block):
                    for j in range(m_block):
                        U_adj[(row_offset + i) * m + (col_offset_A + j), :] = Ub[i * m_block + j, :]
                for j in range(m_block):
                    for k in range(p_block):
                        V_adj[(col_offset_A + j) * p + (col_offset_B + k), :] = Vb[j * p_block + k, :]
                for k in range(p_block):
                    for i in range(n_block):
                        W_adj[(col_offset_B + k) * n + (row_offset + i), :] = Wb[k * n_block + i, :]
                U_list.append(U_adj)
                V_list.append(V_adj)
                W_list.append(W_adj)
                
            else:
                # General schoolbook
                rank = n_block * m_block * p_block
                Ub = np.zeros((n_block * m_block, rank))
                Vb = np.zeros((m_block * p_block, rank))
                Wb = np.zeros((p_block * n_block, rank))
                col = 0
                for i in range(n_block):
                    for j in range(m_block):
                        for k in range(p_block):
                            Ub[i * m_block + j, col] = 1
                            Vb[j * p_block + k, col] = 1
                            Wb[k * n_block + i, col] = 1
                            col += 1
                U_adj = np.zeros((n * m, rank))
                V_adj = np.zeros((m * p, rank))
                W_adj = np.zeros((p * n, rank))
                for i in range(n_block):
                    for j in range(m_block):
                        U_adj[(row_offset + i) * m + (col_offset_A + j), :] = Ub[i * m_block + j, :]
                for j in range(m_block):
                    for k in range(p_block):
                        V_adj[(col_offset_A + j) * p + (col_offset_B + k), :] = Vb[j * p_block + k, :]
                for k in range(p_block):
                    for i in range(n_block):
                        W_adj[(col_offset_B + k) * n + (row_offset + i), :] = Wb[k * n_block + i, :]
                U_list.append(U_adj)
                V_list.append(V_adj)
                W_list.append(W_adj)
        
        def decompose_block(n_b, m_b, p_b, row_off, colA_off, colB_off):
            """Recursively decompose a block using Strassen where possible."""
            if n_b == 2 and m_b == 2 and p_b == 2:
                add_block(2, 2, 2, row_off, colA_off, colB_off)
            elif n_b == 2 and m_b == 2 and p_b == 1:
                add_block(2, 2, 1, row_off, colA_off, colB_off)
            elif n_b == 1 and m_b == 2 and p_b == 2:
                add_block(1, 2, 2, row_off, colA_off, colB_off)
            elif n_b == 2 and m_b == 1 and p_b == 2:
                add_block(2, 1, 2, row_off, colA_off, colB_off)
            elif n_b == 1 and m_b == 1 and p_b == 1:
                add_block(1, 1, 1, row_off, colA_off, colB_off)
            elif n_b == 3 and m_b == 2 and p_b == 3:
                # Split n=3 into [2,1]
                # (2,2,3) and (1,2,3)
                decompose_block(2, 2, 3, row_off, colA_off, colB_off)
                decompose_block(1, 2, 3, row_off + 2, colA_off, colB_off)
            elif n_b == 2 and m_b == 2 and p_b == 3:
                # Split p=3 into [2,1]
                # (2,2,2) and (2,2,1)
                decompose_block(2, 2, 2, row_off, colA_off, colB_off)
                decompose_block(2, 2, 1, row_off, colA_off, colB_off + 2)
            elif n_b == 1 and m_b == 2 and p_b == 3:
                # Split p=3 into [2,1]
                # (1,2,2) and (1,2,1)
                decompose_block(1, 2, 2, row_off, colA_off, colB_off)
                decompose_block(1, 2, 1, row_off, colA_off, colB_off + 2)
            elif n_b == 3 and m_b == 3 and p_b == 3:
                # Split all dimensions into [2,1]
                # 8 sub-blocks
                decompose_block(2, 2, 2, row_off, colA_off, colB_off)
                decompose_block(2, 2, 1, row_off, colA_off, colB_off + 2)
                decompose_block(2, 1, 2, row_off, colA_off + 2, colB_off)
                decompose_block(2, 1, 1, row_off, colA_off + 2, colB_off + 2)
                decompose_block(1, 2, 2, row_off + 2, colA_off, colB_off)
                decompose_block(1, 2, 1, row_off + 2, colA_off, colB_off + 2)
                decompose_block(1, 1, 2, row_off + 2, colA_off + 2, colB_off)
                decompose_block(1, 1, 1, row_off + 2, colA_off + 2, colB_off + 2)
            elif n_b == 1 and m_b == 2 and p_b == 1:
                add_block(1, 2, 1, row_off, colA_off, colB_off)
            elif n_b == 2 and m_b == 1 and p_b == 1:
                add_block(2, 1, 1, row_off, colA_off, colB_off)
            elif n_b == 1 and m_b == 1 and p_b == 2:
                add_block(1, 1, 2, row_off, colA_off, colB_off)
            else:
                # Fallback: schoolbook
                add_block(n_b, m_b, p_b, row_off, colA_off, colB_off)
        
        # Main block structure:
        # m split: [2, 3], p split: [3, 3]
        # Block (m=2, p=3) at m_offset=0, p_offset=0
        decompose_block(3, 2, 3, 0, 0, 0)
        # Block (m=2, p=3) at m_offset=0, p_offset=3
        decompose_block(3, 2, 3, 0, 0, 3)
        # Block (m=3, p=3) at m_offset=2, p_offset=0
        decompose_block(3, 3, 3, 0, 2, 0)
        # Block (m=3, p=3) at m_offset=2, p_offset=3
        decompose_block(3, 3, 3, 0, 2, 3)
        
        if not U_list:
            return None
        
        U_full = np.hstack(U_list)
        V_full = np.hstack(V_list)
        W_full = np.hstack(W_list)
        
        return U_full, V_full, W_full
    
    # Try the block decomposition
    if time.time() < deadline:
        try:
            U_b, V_b, W_b = construct_block_decomposition()
            if U_b is not None:
                rank_b = U_b.shape[1]
                if verify(U_b, V_b, W_b, n, m, p):
                    if rank_b < best_rank:
                        best_result = {'U': U_b.tolist(), 'V': V_b.tolist(), 'W': W_b.tolist()}
                        best_rank = rank_b
        except Exception:
            pass
    
    # Try a Winograd-style decomposition for the 3x5x6 problem
    # Key idea: use the 4x4x4 Strassen-based rank-16 decomposition for a 4x4x4 core
    # and handle the remaining dimensions with schoolbook terms.
    # 
    # Split: n=3 into [2,1], m=5 into [4,1], p=6 into [4,2]
    # Blocks:
    #   (2,4,4): use rank-16 4x4x4 on the 2x4x4 part? No, need 2x4x4.
    #   Actually, let's think differently.
    #
    # Alternative: Use the fact that for 4x4x4, Strassen gives rank 16.
    # For 3x5x6, we can embed into 4x6x6 and use a 4x6x6 decomposition.
    #
    # Let's try a direct approach: use the known rank-16 for 4x4x4 and combine
    # with schoolbook for the remaining parts.
    #
    # Split m=5 into [4,1] and p=6 into [4,2]:
    #   Block (3,4,4): need rank for 3x4x4
    #   Block (3,4,2): need rank for 3x4x2  
    #   Block (3,1,4): rank 12 (schoolbook)
    #   Block (3,1,2): rank 6 (schoolbook)
    #
    # For (3,4,4): split n=3 into [2,1]
    #   (2,4,4): can use 4x4x4 rank-16 if we pad to 4x4x4? No.
    #   Actually (2,4,4) has schoolbook rank 32.
    #   (1,4,4): rank 16
    #
    # This doesn't seem to help much. Let me try a different approach.
    #
    # Use the 5x5x5 Winograd decomposition. The 5x5x5 matrix multiplication
    # can be done with rank 19 using the Winograd algorithm (based on the
    # 5-point DFT/FFT structure).
    #
    # For 3x5x6: pad to 5x5x6, then handle the extra rows/cols.
    # Actually, let's just try to construct a good decomposition directly.
    
    # Try a greedy/local optimization approach starting from the block decomposition
    if time.time() < deadline:
        try:
            # Start from the block decomposition result
            if best_rank < n * m * p:
                U_cur = np.array(best_result['U'], dtype=float)
                V_cur = np.array(best_result['V'], dtype=float)
                W_cur = np.array(best_result['W'], dtype=float)
                r_cur = U_cur.shape[1]
                
                # Try to reduce rank by finding linear dependencies
                # and eliminating redundant terms
                # This is a simple local search: try to zero out one column
                # and see if the decomposition still works.
                
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for col_to_remove in range(r_cur - 1, -1, -1):
                        if time.time() > deadline:
                            break
                        # Try removing this column
                        mask = [i for i in range(r_cur) if i != col_to_remove]
                        U_test = U_cur[:, mask]
                        V_test = V_cur[:, mask]
                        W_test = W_cur[:, mask]
                        
                        # Quick verification
                        valid = True
                        for i in range(n):
                            if not valid:
                                break
                            for j in range(m):
                                if not valid:
                                    break
                                for k in range(p):
                                    s = 0.0
                                    for c in range(len(mask)):
                                        s += U_test[i * m + j, c] * V_test[j * p + k, c] * W_test[k * n + i, c]
                                    if abs(s - 1.0) > 1e-9:
                                        valid = False
                                        break
                        
                        if valid:
                            U_cur = U_test
                            V_cur = V_test
                            W_cur = W_test
                            r_cur = len(mask)
                            improved = True
                            if r_cur < best_rank:
                                best_rank = r_cur
                                best_result = {'U': U_cur.tolist(), 'V': V_cur.tolist(), 'W': W_cur.tolist()}
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
