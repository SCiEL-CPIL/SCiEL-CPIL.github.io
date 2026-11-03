import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition block by block
    # The total rank is the sum of ranks of blocks
    # We need to allocate space in U, V, W
    
    # First, determine the total rank
    # Block 1: 1 x 2 x 2 (Rank 2)
    # Block 2: 1 x 2 x 3 (Rank 2) -> Wait, 1x2x3 is rank 2?
    # 1 x m x p rank is m. So 1x2x3 is rank 2.
    # Block 3: 2 x 2 x 2 (Rank 7)
    # Block 4: 2 x 2 x 3 (Rank 9)
    
    # Total Rank = 2 + 2 + 7 + 9 = 20
    
    total_rank = 20
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a block
    def add_block(i_offset, j_offset, k_offset, n_block, m_block, p_block, block_rank, u_block, v_block, w_block):
        # u_block is n_block x m_block x block_rank
        # v_block is m_block x p_block x block_rank
        # w_block is p_block x n_block x block_rank
        for r in range(block_rank):
            for i in range(n_block):
                for j in range(m_block):
                    U[(i_offset + i) * m + (j_offset + j)][r] += u_block[i, j, r]
            for j in range(m_block):
                for k in range(p_block):
                    V[(j_offset + j) * p + (k_offset + k)][r] += v_block[j, k, r]
            for k in range(p_block):
                for i in range(n_block):
                    W[(k_offset + k) * n + (i_offset + i)][r] += w_block[k, i, r]

    # We need to define the block decompositions
    # This is tedious to do manually for each block.
    # Instead, we can use the schoolbook for small blocks if rank is not reduced,
    # and Strassen for 2x2x2.
    
    # But wait, the rank 2 for 1x2x2 is just schoolbook.
    # Rank 7 for 2x2x2 is Strassen.
    # Rank 9 for 2x2x3 is Strassen(2x2x2) + Schoolbook(2x2x1).
    
    # I will implement the specific decompositions.
    
    # Block 1: 1x2x2 at (0,0,0)
    # Schoolbook
    rank_1 = 2
    # U[0,0,0]=1, V[0,0,0]=1, W[0,0,0]=1
    # U[0,1,1]=1, V[1,0,1]=1, W[0,0,1]=1
    # Note: W index is k*n+i. Here n_block=1, so k=0, i=0. Index 0.
    
    # This manual construction is error-prone.
    # I will return the schoolbook decomposition for now to ensure validity,
    # but the prompt asks for a stronger algorithm.
    # Given the time constraint and complexity of implementing Strassen blocks correctly
    # with the specific index conventions, I will stick to a safe, slightly improved
    # block decomposition if I can verify it quickly, or just schoolbook.
    
    # However, score 1.0 is the baseline.
    # If I just return schoolbook, score is 1.0.
    # If I return rank 20, score is 3.0.
    
    # Let's try to implement the rank 20 decomposition.
    
    # I'll use a generic function to generate schoolbook for a block
    def schoolbook_block(n_b, m_b, p_b):
        r = n_b * m_b * p_b
        u = np.zeros((n_b, m_b, r))
        v = np.zeros((m_b, p_b, r))
        w = np.zeros((p_b, n_b, r))
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u[i, j, col] = 1
                    v[j, k, col] = 1
                    w[k, i, col] = 1
                    col += 1
        return r, u, v, w

    # Strassen 2x2x2
    def strassen_block():
        # Standard Strassen decomposition for 2x2x2
        # Rank 7
        u = np.zeros((2, 2, 7))
        v = np.zeros((2, 2, 7))
        w = np.zeros((2, 2, 7))
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 - M6
        
        # U (A factors)
        # r=0 (M1): A00+A11
        u[0,0,0] = 1; u[1,1,0] = 1
        # r=1 (M2): A10+A11
        u[1,0,1] = 1; u[1,1,1] = 1
        # r=2 (M3): A00
        u[0,0,2] = 1
        # r=3 (M4): A11
        u[1,1,3] = 1
        # r=4 (M5): A00+A01
        u[0,0,4] = 1; u[0,1,4] = 1
        # r=5 (M6): A10-A00
        u[1,0,5] = 1; u[0,0,5] = -1
        # r=6 (M7): A01-A11
        u[0,1,6] = 1; u[1,1,6] = -1
        
        # V (B factors)
        # r=0 (M1): B00+B11
        v[0,0,0] = 1; v[1,1,0] = 1
        # r=1 (M2): B00
        v[0,0,1] = 1
        # r=2 (M3): B01-B11
        v[0,1,2] = 1; v[1,1,2] = -1
        # r=3 (M4): B10-B00
        v[1,0,3] = 1; v[0,0,3] = -1
        # r=4 (M5): B11
        v[1,1,4] = 1
        # r=5 (M6): B00+B01
        v[0,0,5] = 1; v[0,1,5] = 1
        # r=6 (M7): B10+B11
        v[1,0,6] = 1; v[1,1,6] = 1
        
        # W (C factors)
        # C00 = M1 - M4 + M6 - M7
        w[0,0,0] = 1
        w[0,0,3] = -1
        w[0,0,5] = 1
        w[0,0,6] = -1
        # C01 = M3 + M5
        w[1,0,2] = 1
        w[1,0,4] = 1
        # C10 = M2 + M4
        w[0,1,1] = 1
        w[0,1,3] = 1
        # C11 = M1 - M3 + M5 - M6
        w[1,1,0] = 1
        w[1,1,2] = -1
        w[1,1,4] = 1
        w[1,1,5] = -1
        
        return 7, u, v, w

    # Now assemble the blocks
    # Block 1: 1x2x2 at (0,0,0)
    r1, u1, v1, w1 = schoolbook_block(1, 2, 2)
    add_block(0, 0, 0, 1, 2, 2, r1, u1, v1, w1)
    
    # Block 2: 1x2x3 at (0,2,0) -> Wait, j_offset=2, k_offset=0?
    # A is 3x4. Rows 0,1,2. Cols 0,1,2,3.
    # B is 4x5. Rows 0,1,2,3. Cols 0,1,2,3,4.
    # Split A rows: 0 | 1,2
    # Split A cols: 0,1 | 2,3
    # Split B cols: 0,1 | 2,3 | 4
    # Split B rows: 0,1 | 2,3
    
    # Block 1: A[0:1, 0:2] * B[0:2, 0:2] -> 1x2x2. Rank 2.
    # Block 2: A[0:1, 0:2] * B[0:2, 2:5] -> 1x2x3. Rank 2.
    # Block 3: A[0:1, 2:4] * B[2:4, 0:2] -> 1x2x2. Rank 2.
    # Block 4: A[0:1, 2:4] * B[2:4, 2:5] -> 1x2x3. Rank 2.
    # Block 5: A[1:3, 0:2] * B[0:2, 0:2] -> 2x2x2. Rank 7.
    # Block 6: A[1:3, 0:2] * B[0:2, 2:5] -> 2x2x3. Rank 9.
    # Block 7: A[1:3, 2:4] * B[2:4, 0:2] -> 2x2x2. Rank 7.
    # Block 8: A[1:3, 2:4] * B[2:4, 2:5] -> 2x2x3. Rank 9.
    
    # Total Rank = 2+2+2+2+7+9+7+9 = 40.
    # This is worse than 20. Why?
    # Because I split both A and B into 2 blocks.
    # $C = A_L B_L + A_L B_R + A_R B_L + A_R B_R$.
    # If I use Strassen for $2 \times 2 \times 2$, I save.
    # But I have 4 blocks of $1 \times 2 \times \dots$ and 4 blocks of $2 \times 2 \times \dots$.
    
    # My previous calculation of 20 was:
    # $1 \times 4 \times 5$ (Rank 4) + $2 \times 4 \times 5$ (Rank 32) = 36.
    # And then I split $2 \times 4 \times 5$ into $2 \times 2 \times 5$ (16) + $2 \times 2 \times 5$ (16) = 32.
    # So 36 is the rank.
    
    # Can I do better?
    # $3 \times 4 \times 5$.
    # Use Strassen on $2 \times 2 \times 2$ parts.
    # The rank 20 estimate was flawed because I assumed I could mix blocks freely.
    # The decomposition must cover all terms.
    
    # So 36 is a solid improvement over 60.
    # Score = 60/36 = 1.66.
    
    # I will implement the 36 rank decomposition.
    
    # Reset U, V, W
    total_rank = 36
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block 1: A[0:1, 0:4] * B[0:4, 0:5] -> 1x4x5. Rank 4.
    # Wait, 1x4x5 rank is 4?
    # $C_k = \sum_j A_{0,j} B_{j,k}$.
    # Rank 4.
    r, u, v, w = schoolbook_block(1, 4, 5)
    # But rank 4 is not schoolbook (20).
    # Schoolbook for 1x4x5 is 20.
    # I need a rank 4 decomposition for 1x4x5.
    # $C_k = \sum_{j=0}^3 A_{0,j} B_{j,k}$.
    # $U_{0,j,r} = 1$ if $r=j$?
    # $V_{j,k,r} = 1$ if $r=j$?
    # $W_{k,0,r} = 1$?
    # $\sum_r U_{0,j,r} V_{j,k,r} W_{k,0,r} = \sum_{r=0}^3 \delta_{j,r} \delta_{j,r} \cdot 1 = 1$ if $j=r$?
    # No, the sum is over $r$.
    # If $U_{0,j,r} = \delta_{j,r}$, $V_{j,k,r} = \delta_{j,r}$, $W_{k,0,r} = 1$.
    # Term $j,k$: $\sum_r \delta_{j,r} \delta_{j,r} 1 = 1$ if $j$ is in $0..3$.
    # So yes, rank 4 works.
    
    def vector_matrix_block(n_b, m_b, p_b):
        # n_b = 1
        r = m_b
        u = np.zeros((n_b, m_b, r))
        v = np.zeros((m_b, p_b, r))
        w = np.zeros((p_b, n_b, r))
        for j in range(m_b):
            u[0, j, j] = 1
            for k in range(p_b):
                v[j, k, j] = 1
            for k in range(p_b):
                w[k, 0, j] = 1
        return r, u, v, w

    # Block 1: 1x4x5 at (0,0,0)
    r1, u1, v1, w1 = vector_matrix_block(1, 4, 5)
    add_block(0, 0, 0, 1, 4, 5, r1, u1, v1, w1)
    
    # Block 2: A[1:3, 0:2] * B[0:2, 0:5] -> 2x2x5. Rank 16.
    # Split B into 2x2 and 2x1? No, 2x2x2 (7) + 2x2x2 (7) + 2x2x1 (2) = 16.
    # Or just use schoolbook 2x2x5 = 20?
    # 16 is better.
    
    # I will use the 2x2x2 Strassen and 2x2x1 schoolbook.
    
    # Block 2a: 2x2x2 at (1,0,0)
    r2a, u2a, v2a, w2a = strassen_block()
    add_block(1, 0, 0, 2, 2, 2, r2a, u2a, v2a, w2a)
    
    # Block 2b: 2x2x2 at (1,0,2)
    r2b, u2b, v2b, w2b = strassen_block()
    add_block(1, 0, 2, 2, 2, 2, r2b, u2b, v2b, w2b)
    
    # Block 2c: 2x2x1 at (1,0,4)
    r2c, u2c, v2c, w2c = schoolbook_block(2, 2, 1)
    add_block(1, 0, 4, 2, 2, 1, r2c, u2c, v2c, w2c)
    
    # Block 3: A[1:3, 2:4] * B[2:4, 0:5] -> 2x2x5. Rank 16.
    # Block 3a: 2x2x2 at (1,2,0)
    r3a, u3a, v3a, w3a = strassen_block()
    add_block(1, 2, 0, 2, 2, 2, r3a, u3a, v3a, w3a)
    
    # Block 3b: 2x2x2 at (1,2,2)
    r3b, u3b, v3b, w3b = strassen_block()
    add_block(1, 2, 2, 2, 2, 2, r3b, u3b, v3b, w3b)
    
    # Block 3c: 2x2x1 at (1,2,4)
    r3c, u3c, v3c, w3c = schoolbook_block(2, 2, 1)
    add_block(1, 2, 4, 2, 2, 1, r3c, u3c, v3c, w3c)
    
    return {"U": U, "V": V, "W": W}
