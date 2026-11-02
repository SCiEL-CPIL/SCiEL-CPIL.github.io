import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition by summing decompositions of sub-blocks.
    # Block 1: A[0:2, 0:2] * B[0:2, :] -> C[0:2, :]
    # Block 2: A[0:2, 2:4] * B[2:4, :] -> C[0:2, :]
    # Block 3: A[2:3, 0:2] * B[0:2, :] -> C[2:3, :]
    # Block 4: A[2:3, 2:4] * B[2:4, :] -> C[2:3, :]
    
    # Trivial decomposition for a block of size n_b x m_b x p_b
    # Rank is n_b * m_b * p_b
    # We need to map local indices to global indices.
    
    # Total rank estimate:
    # B1: 2*2*7 = 28
    # B2: 2*2*7 = 28
    # B3: 1*2*7 = 14
    # B4: 1*2*7 = 14
    # Total: 84? 
    # Wait, 28+28+14+14 = 84.
    # This is the same as the original!
    
    # Why? Because the original is just the trivial decomposition of the whole 3x4x7.
    # 3*4*7 = 84.
    # My block decomposition sums the trivial decompositions of the parts.
    # Since the parts partition the indices, the total rank is the sum of the ranks.
    # So this approach yields NO improvement if using trivial decompositions for sub-blocks.
    
    # To improve, we MUST use a better decomposition for the sub-blocks.
    # The directive mentions "Strassen-like identities" for 2x2x7.
    # R(2x2x2) = 7.
    # We can decompose 2x2x7 into three 2x2x2 and one 2x2x1.
    # Rank = 3*7 + 1*2 = 23.
    # This is less than 28.
    # Improvement per 2x2x7 block: 28 - 23 = 5.
    # Two such blocks: 10 reduction.
    # For 1x2x7 blocks:
    # R(1x2x7) = 14 (trivial).
    # Can we do better?
    # R(1x2x2) = 4? (Trivial 1*2*2=4).
    # Is there a better one?
    # C_k = A1 B1k + A2 B2k.
    # This is essentially two independent 1x1x2 multiplications? No.
    # If we split k into pairs:
    # 1x2x2. Rank 4.
    # 1x2x1. Rank 2.
    # Total 3*4 + 1*2 = 14.
    # No improvement for 1x2x7.
    
    # So total rank:
    # 2 * 23 (for 2x2x7 blocks) + 2 * 14 (for 1x2x7 blocks)
    # = 46 + 28 = 74.
    # 74 < 84.
    
    # Let's implement this.
    
    def get_strassen_2x2x2():
        # Returns U, V, W for 2x2x2 multiplication, rank 7.
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Indices:
        # U: i*m+j (0..3)
        # V: j*p+k (0..3)
        # W: k*n+i (0..3)
        
        # Standard Strassen decomposition
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Col 0: M1
        # U: A11, A22 -> idx 0, 3
        U[0][0] = 1
        U[3][0] = 1
        # V: B11, B22 -> idx 0, 3
        V[0][0] = 1
        V[3][0] = 1
        # W: C11, C22 -> idx 0, 3
        W[0][0] = 1
        W[3][0] = 1
        
        # Col 1: M2
        # U: A21, A22 -> idx 2, 3
        U[2][1] = 1
        U[3][1] = 1
        # V: B11 -> idx 0
        V[0][1] = 1
        # W: C21, C22 -> idx 2, 3
        W[2][1] = 1
        W[3][1] = -1
        
        # Col 2: M3
        # U: A11 -> idx 0
        U[0][2] = 1
        # V: B12, B22 -> idx 1, 3
        V[1][2] = 1
        V[3][2] = -1
        # W: C12, C22 -> idx 1, 3
        W[1][2] = 1
        W[3][2] = 1
        
        # Col 3: M4
        # U: A22 -> idx 3
        U[3][3] = 1
        # V: B21, B11 -> idx 2, 0
        V[2][3] = 1
        V[0][3] = -1
        # W: C11, C21 -> idx 0, 2
        W[0][3] = 1
        W[2][3] = 1
        
        # Col 4: M5
        # U: A11, A12 -> idx 0, 1
        U[0][4] = 1
        U[1][4] = 1
        # V: B22 -> idx 3
        V[3][4] = 1
        # W: C11, C12 -> idx 0, 1
        W[0][4] = -1
        W[1][4] = 1
        
        # Col 5: M6
        # U: A21, A11 -> idx 2, 0
        U[2][5] = 1
        U[0][5] = -1
        # V: B11, B12 -> idx 0, 1
        V[0][5] = 1
        V[1][5] = 1
        # W: C22 -> idx 3
        W[3][5] = 1
        
        # Col 6: M7
        # U: A12, A22 -> idx 1, 3
        U[1][6] = 1
        U[3][6] = -1
        # V: B21, B22 -> idx 2, 3
        V[2][6] = 1
        V[3][6] = 1
        # W: C11 -> idx 0
        W[0][6] = 1
        
        return U, V, W

    def get_trivial_2x2x1():
        # Rank 2
        U = [[1, 0], [0, 0], [0, 0], [0, 1]]
        V = [[1, 0], [0, 0], [0, 1], [0, 0]]
        W = [[1, 0], [0, 0], [0, 0], [0, 1]]
        return U, V, W

    def get_trivial_1x2x2():
        # Rank 4
        # C1 = A1 B11 + A2 B21
        # C2 = A1 B12 + A2 B22
        # Terms:
        # 1: A1 B11 -> C1
        # 2: A2 B21 -> C1
        # 3: A1 B12 -> C2
        # 4: A2 B22 -> C2
        
        U = [[1, 0, 1, 0], [0, 1, 0, 1]] # A1, A2
        V = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] # B11, B21, B12, B22
        W = [[1, 1, 0, 0], [0, 0, 1, 1]] # C1, C2
        return U, V, W

    def get_trivial_1x2x1():
        # Rank 2
        U = [[1, 0], [0, 1]]
        V = [[1, 0], [0, 1]]
        W = [[1, 1]]
        return U, V, W

    # Assemble 2x2x7
    # Split k=0..6 into (0,1), (2,3), (4,5), (6)
    # 3 blocks of 2x2x2, 1 block of 2x2x1
    
    # Assemble 1x2x7
    # Split k=0..6 into (0,1), (2,3), (4,5), (6)
    # 3 blocks of 1x2x2, 1 block of 1x2x1

    # We need to map these local decompositions to the global 3x4x7 structure.
    
    # Global U: 3*4 = 12 rows.
    # Global V: 4*7 = 28 rows.
    # Global W: 7*3 = 21 rows.
    
    # We will accumulate columns.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a block decomposition
    # block_n, block_m, block_p
    # offset_i (row in A, row in C)
    # offset_j (col in A, row in B)
    # offset_k (col in C, col in B)
    
    def add_block(block_type, off_i, off_j, off_k):
        if block_type == '2x2x2':
            u, v, w = get_strassen_2x2x2()
            # u: 4x7, v: 4x7, w: 4x7
            for r in range(7):
                u_col = [0]*12
                v_col = [0]*28
                w_col = [0]*21
                
                # Map U indices
                # Local i,j -> Global i,j
                # Local i=0,1; j=0,1
                # Global i = off_i + i, j = off_j + j
                for li in range(2):
                    for lj in range(2):
                        li_j = li*2 + lj
                        gi_j = (off_i + li)*4 + (off_j + lj)
                        u_col[gi_j] = u[li_j][r]
                        
                # Map V indices
                # Local j,k -> Global j,k
                # Local j=0,1; k=0,1
                # Global j = off_j + j, k = off_k + k
                for lj in range(2):
                    for lk in range(2):
                        lj_k = lj*2 + lk
                        gj_k = (off_j + lj)*7 + (off_k + lk)
                        v_col[gj_k] = v[lj_k][r]
                        
                # Map W indices
                # Local k,i -> Global k,i
                # Local k=0,1; i=0,1
                # Global k = off_k + k, i = off_i + i
                for lk in range(2):
                    for li in range(2):
                        lk_i = lk*2 + li
                        gk_i = (off_k + lk)*3 + (off_i + li)
                        w_col[gk_i] = w[lk_i][r]
                        
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
                
        elif block_type == '2x2x1':
            u, v, w = get_trivial_2x2x1()
            # u: 4x2, v: 4x2, w: 4x2
            for r in range(2):
                u_col = [0]*12
                v_col = [0]*28
                w_col = [0]*21
                
                for li in range(2):
                    for lj in range(2):
                        li_j = li*2 + lj
                        gi_j = (off_i + li)*4 + (off_j + lj)
                        u_col[gi_j] = u[li_j][r]
                        
                for lj in range(2):
                    for lk in range(1):
                        lj_k = lj*2 + lk
                        gj_k = (off_j + lj)*7 + (off_k + lk)
                        v_col[gj_k] = v[lj_k][r]
                        
                for lk in range(1):
                    for li in range(2):
                        lk_i = lk*2 + li
                        gk_i = (off_k + lk)*3 + (off_i + li)
                        w_col[gk_i] = w[lk_i][r]
                        
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
                
        elif block_type == '1x2x2':
            u, v, w = get_trivial_1x2x2()
            # u: 2x4, v: 4x4, w: 2x4
            for r in range(4):
                u_col = [0]*12
                v_col = [0]*28
                w_col = [0]*21
                
                for li in range(1):
                    for lj in range(2):
                        li_j = li*2 + lj
                        gi_j = (off_i + li)*4 + (off_j + lj)
                        u_col[gi_j] = u[li_j][r]
                        
                for lj in range(2):
                    for lk in range(2):
                        lj_k = lj*2 + lk
                        gj_k = (off_j + lj)*7 + (off_k + lk)
                        v_col[gj_k] = v[lj_k][r]
                        
                for lk in range(2):
                    for li in range(1):
                        lk_i = lk*2 + li
                        gk_i = (off_k + lk)*3 + (off_i + li)
                        w_col[gk_i] = w[lk_i][r]
                        
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
                
        elif block_type == '1x2x1':
            u, v, w = get_trivial_1x2x1()
            # u: 2x2, v: 2x2, w: 2x2
            for r in range(2):
                u_col = [0]*12
                v_col = [0]*28
                w_col = [0]*21
                
                for li in range(1):
                    for lj in range(2):
                        li_j = li*2 + lj
                        gi_j = (off_i + li)*4 + (off_j + lj)
                        u_col[gi_j] = u[li_j][r]
                        
                for lj in range(2):
                    for lk in range(1):
                        lj_k = lj*2 + lk
                        gj_k = (off_j + lj)*7 + (off_k + lk)
                        v_col[gj_k] = v[lj_k][r]
                        
                for lk in range(1):
                    for li in range(1):
                        lk_i = lk*2 + li
                        gk_i = (off_k + lk)*3 + (off_i + li)
                        w_col[gk_i] = w[lk_i][r]
                        
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)

    # Block 1: 2x2x7 at off_i=0, off_j=0
    # k=0,1 -> 2x2x2
    add_block('2x2x2', 0, 0, 0)
    # k=2,3 -> 2x2x2
    add_block('2x2x2', 0, 0, 2)
    # k=4,5 -> 2x2x2
    add_block('2x2x2', 0, 0, 4)
    # k=6 -> 2x2x1
    add_block('2x2x1', 0, 0, 6)
    
    # Block 2: 2x2x7 at off_i=0, off_j=2
    add_block('2x2x2', 0, 2, 0)
    add_block('2x2x2', 0, 2, 2)
    add_block('2x2x2', 0, 2, 4)
    add_block('2x2x1', 0, 2, 6)
    
    # Block 3: 1x2x7 at off_i=2, off_j=0
    add_block('1x2x2', 2, 0, 0)
    add_block('1x2x2', 2, 0, 2)
    add_block('1x2x2', 2, 0, 4)
    add_block('1x2x1', 2, 0, 6)
    
    # Block 4: 1x2x7 at off_i=2, off_j=2
    add_block('1x2x2', 2, 2, 0)
    add_block('1x2x2', 2, 2, 2)
    add_block('1x2x2', 2, 2, 4)
    add_block('1x2x1', 2, 2, 6)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
