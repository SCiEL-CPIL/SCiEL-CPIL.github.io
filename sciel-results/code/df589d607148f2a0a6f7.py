import numpy as np
import time
import sys

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor T[i, j, k] = 1 if using row i of A, col j of A, row k of B
    # Actually, the tensor for multiplying A (n x m) by B (m x p) to get C (n x p)
    # T[i, j, k] should be 1 if i < n, j < m, k < p
    # The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Wait, let's re-check the convention.
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # The product is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all valid i, j, k.
    
    # Let's construct the tensor directly.
    # Tensor shape: (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    # Note that j is shared between a and b, and k is shared between b and c, and i is shared between c and a.
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Compute the decomposition tensor
    # D = U @ V.T @ W.T? No.
    # sum_c U[a, c] * V[b, c] * W[c_idx, c]
    # This is the elementwise product sum over c.
    
    # We can compute this as:
    # D[a, b, c_idx] = sum_c U[a, c] * V[b, c] * W[c_idx, c]
    # This is a tensor contraction.
    # We can do: for each c, outer product of U[:, c], V[:, c], W[:, c]
    
    D = np.zeros_like(T)
    for c in range(rank):
        D += np.einsum('a,b,ab->a', U[:, c], V[:, c], np.ones((1,))) # This is wrong.
        
    # Correct way:
    # D[a, b, c_idx] = sum_k U[a, k] * V[b, k] * W[c_idx, k]
    # Let's use einsum: 'ak,bk,ck->abc'
    D = np.einsum('ak,bk,ck->abc', U, V, W)
    
    return np.allclose(D, T, atol=1e-9)

def construct():
    n, m, p = 2, 5, 6
    
    # We'll try to construct a rank 54 decomposition using Strassen for 2x2x2 blocks.
    # Split m=5 into 2+2+1.
    # A = [A1 (2x2), A2 (2x2), A3 (2x1)]
    # B = [B1 (2x6); B2 (2x6); B3 (1x6)]
    
    # For each block A_x B_x:
    # B_x is (2x6) or (1x6). We can split B_x into three 2x2 or 1x2 blocks.
    # B1 = [B11, B12, B13] where each is 2x2.
    # B2 = [B21, B22, B23] where each is 2x2.
    # B3 = [B31, B32, B33] where each is 1x2.
    
    # C = A1 B1 + A2 B2 + A3 B3
    # C = A1(B11+B12+B13) + A2(B21+B22+B23) + A3(B31+B32+B33)
    
    # Each A_x B_y is a small matrix multiplication.
    # A1 B11 is 2x2 * 2x2 -> 2x2. Rank 7 (Strassen).
    # A1 B12 is 2x2 * 2x2 -> 2x2. Rank 7.
    # A1 B13 is 2x2 * 2x2 -> 2x2. Rank 7.
    # A2 B21 is 2x2 * 2x2 -> 2x2. Rank 7.
    # A2 B22 is 2x2 * 2x2 -> 2x2. Rank 7.
    # A2 B23 is 2x2 * 2x2 -> 2x2. Rank 7.
    # A3 B31 is 2x1 * 1x2 -> 2x2. Rank 4 (Schoolbook).
    # A3 B32 is 2x1 * 1x2 -> 2x2. Rank 4.
    # A3 B33 is 2x1 * 1x2 -> 2x2. Rank 4.
    
    # Total rank: 6*7 + 3*4 = 42 + 12 = 54.
    
    # We need to map these to the global U, V, W indices.
    # Let's create helper functions to build U, V, W for each block.
    
    # Strassen for 2x2x2:
    # A = [[a, b], [c, d]], B = [[x, y], [z, w]]
    # m1 = (a+d)(x+w)
    # m2 = (c+d)x
    # m3 = a(y-w)
    # m4 = d(z-x)
    # m5 = (a+b)w
    # m6 = (c-a)(x+z)
    # m7 = (b-d)(y+w)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        # Indices:
        # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1 = (a+d)(x+w)
        # a=A[0,0]=0, d=A[1,1]=3, x=B[0,0]=0, w=B[1,1]=3
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C11 += m1, C22 += m1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2 = (c+d)x
        # c=A[1,0]=2, d=A[1,1]=3, x=B[0,0]=0
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C21 += m2
        W[2, 1] = 1
        
        # m3 = a(y-w)
        # a=A[0,0]=0, y=B[0,1]=1, w=B[1,1]=3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C12 += m3, C22 += m3
        W[1, 2] = 1; W[3, 2] = 1
        
        # m4 = d(z-x)
        # d=A[1,1]=3, z=B[1,0]=2, x=B[0,0]=0
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C11 += m4, C21 += m4
        W[0, 3] = 1; W[2, 3] = 1
        
        # m5 = (a+b)w
        # a=A[0,0]=0, b=A[0,1]=1, w=B[1,1]=3
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C11 -= m5, C12 += m5
        W[0, 4] = -1; W[1, 4] = 1
        
        # m6 = (c-a)(x+z)
        # c=A[1,0]=2, a=A[0,0]=0, x=B[0,0]=0, z=B[1,0]=2
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        # C22 += m6
        W[3, 5] = 1
        
        # m7 = (b-d)(y+w)
        # b=A[0,1]=1, d=A[1,1]=3, y=B[0,1]=1, w=B[1,1]=3
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        # C11 += m7
        W[0, 6] = 1
        
        return U, V, W
    
    def schoolbook_2x1x2():
        """Return U, V, W for 2x1x2 schoolbook decomposition."""
        # A is 2x1, B is 1x2, C is 2x2
        # A indices: (0,0)=0, (1,0)=1
        # B indices: (0,0)=0, (0,1)=1
        # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        
        # C = [[A0*B0, A0*B1], [A1*B0, A1*B1]]
        
        # C[0,0] = A[0,0]*B[0,0]
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # C[0,1] = A[0,0]*B[0,1]
        U[0, 1] = 1
        V[1, 1] = 1
        W[1, 1] = 1
        
        # C[1,0] = A[1,0]*B[0,0]
        U[1, 2] = 1
        V[0, 2] = 1
        W[2, 2] = 1
        
        # C[1,1] = A[1,0]*B[0,1]
        U[1, 3] = 1
        V[1, 3] = 1
        W[3, 3] = 1
        
        return U, V, W
    
    # Now we need to assemble these into the global U, V, W.
    # n=2, m=5, p=6
    # U shape: (10, 54)
    # V shape: (30, 54)
    # W shape: (12, 54)
    
    U_global = np.zeros((10, 54))
    V_global = np.zeros((30, 54))
    W_global = np.zeros((12, 54))
    
    col_offset = 0
    
    # Block 1: A1 (2x2) * B11 (2x2)
    # A1 indices: rows 0,1 of A, cols 0,1 of A
    # B11 indices: rows 0,1 of B, cols 0,1 of B
    # C indices: rows 0,1 of C, cols 0,1 of C
    
    # A index mapping: i*m+j -> i*5+j
    # B index mapping: j*p+k -> j*6+k
    # C index mapping: k*n+i -> k*2+i
    
    # For A1:
    # A[0,0] -> U index 0*5+0 = 0
    # A[0,1] -> U index 0*5+1 = 1
    # A[1,0] -> U index 1*5+0 = 5
    # A[1,1] -> U index 1*5+1 = 6
    
    # For B11:
    # B[0,0] -> V index 0*6+0 = 0
    # B[0,1] -> V index 0*6+1 = 1
    # B[1,0] -> V index 1*6+0 = 6
    # B[1,1] -> V index 1*6+1 = 7
    
    # For C11:
    # C[0,0] -> W index 0*2+0 = 0
    # C[0,1] -> W index 1*2+0 = 2
    # C[1,0] -> W index 0*2+1 = 1
    # C[1,1] -> W index 1*2+1 = 3
    
    # Wait, the W index is k*n+i.
    # C[i,k] corresponds to W index k*n+i.
    # So C[0,0] -> k=0, i=0 -> W index 0
    # C[0,1] -> k=1, i=0 -> W index 2
    # C[1,0] -> k=0, i=1 -> W index 1
    # C[1,1] -> k=1, i=1 -> W index 3
    
    # Let's write a helper to add a block decomposition to the global arrays.
    
    def add_block(U_block, V_block, W_block, U_map, V_map, W_map, offset):
        """Add a block decomposition to the global arrays.
        
        U_block: (n_block*m_block, rank_block)
        V_block: (m_block*p_block, rank_block)
        W_block: (p_block*n_block, rank_block)
        
        U_map: list of global U indices for the block's U rows
        V_map: list of global V indices for the block's V rows
        W_map: list of global W indices for the block's W rows
        """
        for r in range(rank_block):
            for u_idx, u_global_idx in enumerate(U_map):
                U_global[u_global_idx, offset + r] += U_block[u_idx, r]
            for v_idx, v_global_idx in enumerate(V_map):
                V_global[v_global_idx, offset + r] += V_block[v_idx, r]
            for w_idx, w_global_idx in enumerate(W_map):
                W_global[w_global_idx, offset + r] += W_block[w_idx, r]
    
    # We need to be careful with the indices.
    # Let's define the mappings explicitly.
    
    # Block 1: A1 * B11
    # A1: rows 0,1, cols 0,1
    # U_map: [0, 1, 5, 6]
    # B11: rows 0,1, cols 0,1
    # V_map: [0, 1, 6, 7]
    # C11: rows 0,1, cols 0,1
    # W_map: [0, 2, 1, 3]  # k*n+i: (0,0)->0, (1,0)->2, (0,1)->1, (1,1)->3
    
    U_str, V_str, W_str = strassen_2x2x2()
    add_block(U_str, V_str, W_str, [0, 1, 5, 6], [0, 1, 6, 7], [0, 2, 1, 3], col_offset)
    col_offset += 7
    
    # Block 2: A1 * B12
    # A1: rows 0,1, cols 0,1
    # U_map: [0, 1, 5, 6]
    # B12: rows 0,1, cols 2,3
    # V_map: [2, 3, 8, 9]
    # C12: rows 0,1, cols 2,3
    # W_map: [4, 6, 5, 7]  # k*n+i: (2,0)->4, (3,0)->6, (2,1)->5, (3,1)->7
    
    add_block(U_str, V_str, W_str, [0, 1, 5, 6], [2, 3, 8, 9], [4, 6, 5, 7], col_offset)
    col_offset += 7
    
    # Block 3: A1 * B13
    # A1: rows 0,1, cols 0,1
    # U_map: [0, 1, 5, 6]
    # B13: rows 0,1, cols 4,5
    # V_map: [4, 5, 10, 11]
    # C13: rows 0,1, cols 4,5
    # W_map: [8, 10, 9, 11]  # k*n+i: (4,0)->8, (5,0)->10, (4,1)->9, (5,1)->11
    
    add_block(U_str, V_str, W_str, [0, 1, 5, 6], [4, 5, 10, 11], [8, 10, 9, 11], col_offset)
    col_offset += 7
    
    # Block 4: A2 * B21
    # A2: rows 2,3, cols 0,1
    # U_map: [10, 11, 15, 16]
    # B21: rows 2,3, cols 0,1
    # V_map: [12, 13, 18, 19]
    # C21: rows 2,3, cols 0,1
    # W_map: [0, 2, 1, 3]  # Same C indices as before, but different A rows
    
    add_block(U_str, V_str, W_str, [10, 11, 15, 16], [12, 13, 18, 19], [0, 2, 1, 3], col_offset)
    col_offset += 7
    
    # Block 5: A2 * B22
    # A2: rows 2,3, cols 0,1
    # U_map: [10, 11, 15, 16]
    # B22: rows 2,3, cols 2,3
    # V_map: [14, 15, 20, 21]
    # C22: rows 2,3, cols 2,3
    # W_map: [4, 6, 5, 7]
    
    add_block(U_str, V_str, W_str, [10, 11, 15, 16], [14, 15, 20, 21], [4, 6, 5, 7], col_offset)
    col_offset += 7
    
    # Block 6: A2 * B23
    # A2: rows 2,3, cols 0,1
    # U_map: [10, 11, 15, 16]
    # B23: rows 2,3, cols 4,5
    # V_map: [16, 17, 22, 23]
    # C23: rows 2,3, cols 4,5
    # W_map: [8, 10, 9, 11]
    
    add_block(U_str, V_str, W_str, [10, 11, 15, 16], [16, 17, 22, 23], [8, 10, 9, 11], col_offset)
    col_offset += 7
    
    # Now for A3 * B3x blocks.
    # A3: rows 2,3, col 2 (since m=5, cols are 0,1,2,3,4, so A3 is col 2? No, m=5, A is 2x5.
    # We split m=5 into 2+2+1. So A1 is cols 0,1, A2 is cols 2,3, A3 is col 4.
    # Wait, I used cols 0,1 for A1, cols 2,3 for A2? No, the split is on the m dimension.
    # A is 2x5. We split the columns into groups of 2, 2, 1.
    # A1: cols 0,1
    # A2: cols 2,3
    # A3: col 4
    
    # Let's correct the U_map for A2 and A3.
    # A2: rows 2,3? No, A is 2x5. Rows are 0,1. Cols are 0,1,2,3,4.
    # U index is i*m+j = i*5+j.
    # A1: i=0,1; j=0,1. U indices: 0,1,5,6. Correct.
    # A2: i=0,1; j=2,3. U indices: 2,3,7,8.
    # A3: i=0,1; j=4. U indices: 4,9.
    
    # I made a mistake in the U_map for A2. Let's redo it.
    
    U_global = np.zeros((10, 54))
    V_global = np.zeros((30, 54))
    W_global = np.zeros((12, 54))
    col_offset = 0
    
    # Block 1: A1 * B11
    # A1: i=0,1; j=0,1. U_map: [0, 1, 5, 6]
    # B11: j=0,1; k=0,1. V_map: [0, 1, 6, 7]
    # C: i=0,1; k=0,1. W_map: [0, 2, 1, 3]
    
    add_block(U_str, V_str, W_str, [0, 1, 5, 6], [0, 1, 6, 7], [0, 2, 1, 3], col_offset)
    col_offset += 7
    
    # Block 2: A1 * B12
    # A1: i=0,1; j=0,1. U_map: [0, 1, 5, 6]
    # B12: j=0,1; k=2,3. V_map: [2, 3, 8, 9]
    # C: i=0,1; k=2,3. W_map: [4, 6, 5, 7]
    
    add_block(U_str, V_str, W_str, [0, 1, 5, 6], [2, 3, 8, 9], [4, 6, 5, 7], col_offset)
    col_offset += 7
    
    # Block 3: A1 * B13
    # A1: i=0,1; j=0,1. U_map: [0, 1, 5, 6]
    # B13: j=0,1; k=4,5. V_map: [4, 5, 10, 11]
    # C: i=0,1; k=4,5. W_map: [8, 10, 9, 11]
    
    add_block(U_str, V_str, W_str, [0, 1, 5, 6], [4, 5, 10, 11], [8, 10, 9, 11], col_offset)
    col_offset += 7
    
    # Block 4: A2 * B21
    # A2: i=0,1; j=2,3. U_map: [2, 3, 7, 8]
    # B21: j=2,3; k=0,1. V_map: [12, 13, 18, 19]
    # C: i=0,1; k=0,1. W_map: [0, 2, 1, 3]
    
    add_block(U_str, V_str, W_str, [2, 3, 7, 8], [12, 13, 18, 19], [0, 2, 1, 3], col_offset)
    col_offset += 7
    
    # Block 5: A2 * B22
    # A2: i=0,1; j=2,3. U_map: [2, 3, 7, 8]
    # B22: j=2,3; k=2,3. V_map: [14, 15, 20, 21]
    # C: i=0,1; k=2,3. W_map: [4, 6, 5, 7]
    
    add_block(U_str, V_str, W_str, [2, 3, 7, 8], [14, 15, 20, 21], [4, 6, 5, 7], col_offset)
    col_offset += 7
    
    # Block 6: A2 * B23
    # A2: i=0,1; j=2,3. U_map: [2, 3, 7, 8]
    # B23: j=2,3; k=4,5. V_map: [16, 17, 22, 23]
    # C: i=0,1; k=4,5. W_map: [8, 10, 9, 11]
    
    add_block(U_str, V_str, W_str, [2, 3, 7, 8], [16, 17, 22, 23], [8, 10, 9, 11], col_offset)
    col_offset += 7
    
    # Now A3 * B3x
    # A3: i=0,1; j=4. U_map: [4, 9]
    # B31: j=4; k=0,1. V_map: [24, 25]
    # C31: i=0,1; k=0,1. W_map: [0, 2, 1, 3]
    
    U_sb, V_sb, W_sb = schoolbook_2x1x2()
    add_block(U_sb, V_sb, W_sb, [4, 9], [24, 25], [0, 2, 1, 3], col_offset)
    col_offset += 4
    
    # B32: j=4; k=2,3. V_map: [26, 27]
    # C32: i=0,1; k=2,3. W_map: [4, 6, 5, 7]
    
    add_block(U_sb, V_sb, W_sb, [4, 9], [26, 27], [4, 6, 5, 7], col_offset)
    col_offset += 4
    
    # B33: j=4; k=4,5. V_map: [28, 29]
    # C33: i=0,1; k=4,5. W_map: [8, 10, 9, 11]
    
    add_block(U_sb, V_sb, W_sb, [4, 9], [28, 29], [8, 10, 9, 11], col_offset)
    col_offset += 4
    
    # Verify the decomposition
    if not _verify(U_global, V_global, W_global, n, m, p):
        # Fallback to schoolbook
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
    
    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
