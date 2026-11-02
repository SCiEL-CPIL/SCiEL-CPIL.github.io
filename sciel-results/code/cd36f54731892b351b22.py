import time

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Returns a rank-78 decomposition for matrix multiplication M_{3,4,7}
    using a block decomposition with Strassen's algorithm for 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS
    
    # Global dimensions
    U_dim = n * m  # 12
    V_dim = m * p  # 28
    W_dim = p * n  # 21
    
    # We will accumulate the rank
    total_rank = 0
    U = []
    V = []
    W = []
    
    # Helper to add a decomposition block
    def add_block(U_local, V_local, W_local, U_offset, V_offset, W_offset):
        """
        U_local: (n_l * m_l) x r
        V_local: (m_l * p_l) x r
        W_local: (p_l * n_l) x r
        Offsets are (row_start, col_start) for U and V, (row_start, col_start) for W?
        Actually, U is indexed by (i, j) where i is row of A, j is col of A.
        V is indexed by (j, k) where j is row of B, k is col of B.
        W is indexed by (k, i) where k is col of C, i is row of C.
        
        A block A[i1:i2, j1:j2] * B[j1:j2, k1:k2] -> C[i1:i2, k1:k2]
        
        U indices: i in [i1, i2), j in [j1, j2). Index = i*m + j.
        V indices: j in [j1, j2), k in [k1, k2). Index = j*p + k.
        W indices: k in [k1, k2), i in [i1, i2). Index = k*n + i.
        """
        nonlocal total_rank, U, V, W
        
        r = U_local[0].__len__() if U_local else 0
        n_l = len(U_local) // (j2-j1) if (j2-j1)>0 else 0 # This is fragile
        # Better to pass dimensions explicitly or infer from local matrices
        # Let's assume U_local is (n_l*m_l) x r
        # We need to map local indices (i_l, j_l) to global (i, j)
        
        # For this specific problem, we can just hardcode the mappings
        # But a generic helper is better.
        
        pass

    # Given the complexity of generic mapping, and fixed dimensions, 
    # we will construct the blocks explicitly.
    
    # Strassen M222
    U_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    # Transpose to (4, 7)
    U_s = [[U_s[j][i] for j in range(7)] for i in range(4)]
    
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    # Transpose to (4, 7)
    V_s = [[V_s[j][i] for j in range(7)] for i in range(4)]
    
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    # W is already (4, 7)
    
    # Schoolbook MnmP
    def get_schoolbook(n_l, m_l, p_l):
        r = n_l * m_l * p_l
        U_l = [[0]*r for _ in range(n_l*m_l)]
        V_l = [[0]*r for _ in range(m_l*p_l)]
        W_l = [[0]*r for _ in range(p_l*n_l)]
        col = 0
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    U_l[i*m_l+j][col] = 1
                    V_l[j*p_l+k][col] = 1
                    W_l[k*n_l+i][col] = 1
                    col += 1
        return U_l, V_l, W_l

    # Block 1: A11 (2x2) * B11 (2x4) -> C11 (2x4)
    # B11 is two 2x2 blocks: B11a (cols 0,1), B11b (cols 2,3)
    # A11 rows 0,1 cols 0,1.
    # B11a rows 0,1 cols 0,1.
    # B11b rows 0,1 cols 2,3.
    
    # We need to add 2 Strassen blocks.
    
    # Helper to add Strassen block for A[i1:i2, j1:j2] * B[j1:j2, k1:k2]
    def add_strassen(i1, j1, k1):
        # n_l=2, m_l=2, p_l=2
        # U indices: i in [i1, i1+2), j in [j1, j1+2)
        # V indices: j in [j1, j1+2), k in [k1, k1+2)
        # W indices: k in [k1, k1+2), i in [i1, i1+2)
        
        # Map local U (4x7) to global U
        # Local index (i_l, j_l) -> Global (i1+i_l, j1+j_l)
        # Global U index = (i1+i_l)*4 + (j1+j_l)
        
        for i_l in range(2):
            for j_l in range(2):
                u_idx = (i1+i_l)*4 + (j1+j_l)
                for c in range(7):
                    U[u_idx][total_rank + c] += U_s[i_l*2+j_l][c]
                    
        for j_l in range(2):
            for k_l in range(2):
                v_idx = (j1+j_l)*7 + (k1+k_l)
                for c in range(7):
                    V[v_idx][total_rank + c] += V_s[j_l*2+k_l][c]
                    
        for k_l in range(2):
            for i_l in range(2):
                w_idx = (k1+k_l)*3 + (i1+i_l)
                for c in range(7):
                    W[w_idx][total_rank + c] += W_s[k_l*2+i_l][c]
                    
        total_rank += 7

    # Helper to add Schoolbook block
    def add_schoolbook(i1, j1, k1, n_l, m_l, p_l):
        U_l, V_l, W_l = get_schoolbook(n_l, m_l, p_l)
        r = n_l * m_l * p_l
        
        for i_l in range(n_l):
            for j_l in range(m_l):
                u_idx = (i1+i_l)*4 + (j1+j_l)
                for c in range(r):
                    U[u_idx][total_rank + c] += U_l[i_l*m_l+j_l][c]
                    
        for j_l in range(m_l):
            for k_l in range(p_l):
                v_idx = (j1+j_l)*7 + (k1+k_l)
                for c in range(r):
                    V[v_idx][total_rank + c] += V_l[j_l*p_l+k_l][c]
                    
        for k_l in range(p_l):
            for i_l in range(n_l):
                w_idx = (k1+k_l)*3 + (i1+i_l)
                for c in range(r):
                    W[w_idx][total_rank + c] += W_l[k_l*n_l+i_l][c]
                    
        total_rank += r

    # Initialize U, V, W with zeros of size total_rank (we don't know it yet, so we append columns)
    # Actually, we can just create lists of lists and append columns as we go.
    # But we need to update existing columns? No, each block adds new columns.
    
    U = [[0 for _ in range(78)] for _ in range(12)]
    V = [[0 for _ in range(78)] for _ in range(28)]
    W = [[0 for _ in range(78)] for _ in range(21)]
    
    # Block 1: A11 (2x2) * B11a (2x2) -> C11a (2x2)
    # i1=0, j1=0, k1=0
    add_strassen(0, 0, 0)
    
    # Block 2: A11 (2x2) * B11b (2x2) -> C11b (2x2)
    # i1=0, j1=0, k1=2
    add_strassen(0, 0, 2)
    
    # Block 3: A12 (2x2) * B21a (2x2) -> C21a (2x2)
    # A12: rows 0,1, cols 2,3. j1=2.
    # B21a: rows 2,3, cols 0,1. j1=2, k1=0.
    # C21a: rows 0,1, cols 0,1. i1=0, k1=0.
    add_strassen(0, 2, 0)
    
    # Block 4: A12 (2x2) * B21b (2x2) -> C21b (2x2)
    # i1=0, j1=2, k1=2
    add_strassen(0, 2, 2)
    
    # Block 5: A11 (2x2) * B12a (2x2) -> C12a (2x2)
    # B12a: rows 0,1, cols 4,5. k1=4.
    # C12a: rows 0,1, cols 4,5.
    add_strassen(0, 0, 4)
    
    # Block 6: A11 (2x2) * B12b (2x1) -> C12b (2x1)
    # B12b: rows 0,1, col 6. k1=6, p_l=1.
    # Schoolbook M221
    add_schoolbook(0, 0, 6, 2, 2, 1)
    
    # Block 7: A12 (2x2) * B22a (2x2) -> C22a (2x2)
    # B22a: rows 2,3, cols 4,5. j1=2, k1=4.
    add_strassen(0, 2, 4)
    
    # Block 8: A12 (2x2) * B22b (2x1) -> C22b (2x1)
    # B22b: rows 2,3, col 6. j1=2, k1=6.
    add_schoolbook(0, 2, 6, 2, 2, 1)
    
    # Block 9: A21 (1x2) * B11a (2x2) -> C11a (1x2)
    # A21: row 2, cols 0,1. i1=2, j1=0, n_l=1.
    # B11a: rows 0,1, cols 0,1.
    # C11a: row 2, cols 0,1.
    add_schoolbook(2, 0, 0, 1, 2, 2)
    
    # Block 10: A21 (1x2) * B11b (2x2) -> C11b (1x2)
    add_schoolbook(2, 0, 2, 1, 2, 2)
    
    # Block 11: A22 (1x2) * B21a (2x2) -> C21a (1x2)
    # A22: row 2, cols 2,3. j1=2.
    add_schoolbook(2, 2, 0, 1, 2, 2)
    
    # Block 12: A22 (1x2) * B21b (2x2) -> C21b (1x2)
    add_schoolbook(2, 2, 2, 1, 2, 2)
    
    # Block 13: A21 (1x2) * B12a (2x2) -> C12a (1x2)
    add_schoolbook(2, 0, 4, 1, 2, 2)
    
    # Block 14: A21 (1x2) * B12b (2x1) -> C12b (1x1)
    add_schoolbook(2, 0, 6, 1, 2, 1)
    
    # Block 15: A22 (1x2) * B22a (2x2) -> C22a (1x2)
    add_schoolbook(2, 2, 4, 1, 2, 2)
    
    # Block 16: A22 (1x2) * B22b (2x1) -> C22b (1x1)
    add_schoolbook(2, 2, 6, 1, 2, 1)
    
    return {"U": U, "V": V, "W": W}
