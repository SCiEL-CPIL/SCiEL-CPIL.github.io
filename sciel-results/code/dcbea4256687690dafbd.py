"""Program for 4x5 by 5x5 matrix multiplication tensor decomposition.
Uses Strassen-based block decomposition to reduce rank below schoolbook 100.
"""
DIMENSIONS = [4, 5, 5]

def _strassen_2x2():
    """Return the 7 rank-1 tensors for 2x2 by 2x2 matrix multiplication.
    Each tensor is (u, v, w) where u is 2x2, v is 2x2, w is 2x2.
    Indices: u[i,j], v[j,k], w[k,i] for the standard convention.
    Actually for our problem: U has shape (n*m, rank), V (m*p, rank), W (p*n, rank)
    For 2x2*2x2: U is (4, r), V is (4, r), W is (4, r)
    U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    So for 2x2: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrices
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    tensors = []
    # M1 = (A11+A22)(B11+B22) -> C11, C22
    # A indices: (0,0),(1,1); B indices: (0,0),(1,1); C: (0,0),(1,1)
    u1 = [(0,0,1),(1,1,1)]  # A11 + A22
    v1 = [(0,0,1),(1,1,1)]  # B11 + B22
    w1 = [(0,0,1),(1,1,1)]  # C11 + C22
    tensors.append((u1, v1, w1))
    
    # M2 = (A21+A22)B11 -> C21, C22
    u2 = [(1,0,1),(1,1,1)]  # A21 + A22
    v2 = [(0,0,1)]           # B11
    w2 = [(0,1,1),(1,1,-1)]  # C21 - C22
    tensors.append((u2, v2, w2))
    
    # M3 = A11(B21-B22) -> C12, C22
    u3 = [(0,0,1)]           # A11
    v3 = [(1,0,1),(1,1,-1)]  # B21 - B22
    w3 = [(0,1,1),(1,1,1)]   # C12 + C22
    tensors.append((u3, v3, w3))
    
    # M4 = A22(B12-B11) -> C11, C21
    u4 = [(1,1,1)]           # A22
    v4 = [(0,1,1),(0,0,-1)]  # B12 - B11
    w4 = [(0,0,1),(1,0,1)]   # C11 + C21
    tensors.append((u4, v4, w4))
    
    # M5 = (A11+A12)B22 -> C11, C12
    u5 = [(0,0,1),(0,1,1)]   # A11 + A12
    v5 = [(1,1,1)]           # B22
    w5 = [(0,0,-1),(0,1,1)]  # -C11 + C12
    tensors.append((u5, v5, w5))
    
    # M6 = (A21-A11)(B11+B12) -> C22
    u6 = [(1,0,1),(0,0,-1)]  # A21 - A11
    v6 = [(0,0,1),(0,1,1)]   # B11 + B12
    w6 = [(1,1,1)]           # C22
    tensors.append((u6, v6, w6))
    
    # M7 = (A12-A22)(B21+B22) -> C11
    u7 = [(0,1,1),(1,1,-1)]  # A12 - A22
    v7 = [(1,0,1),(1,1,1)]   # B21 + B22
    w7 = [(0,0,1)]           # C11
    tensors.append((u7, v7, w7))
    
    return tensors

def construct():
    """Construct rank-minimal decomposition for 4x5 * 5x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    # We'll build the decomposition by splitting into blocks.
    # A is 4x5, B is 5x5, C is 4x5
    # Split A into rows: A1 (2x5), A2 (2x5)
    # Split B into cols: B1 (5x2), B2 (5x2), B3 (5x1)
    # C = A*B = A1*B1 + A2*B1 ... wait, that's not right.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Split A into top 2 rows (A_top: 2x5) and bottom 2 rows (A_bot: 2x5)
    # Split B into left 2 cols (B_left: 5x2), middle 2 cols (B_mid: 5x2), right 1 col (B_right: 5x1)
    # Then C_top = A_top * B (2x5)
    # C_bot = A_bot * B (2x5)
    # This doesn't reduce rank directly.
    
    # Better approach: Use the known fact that for multiplying an m x n matrix by n x p matrix,
    # the rank is at most the product of the ranks of sub-block decompositions.
    
    # Let's try a different splitting:
    # A: 4x5, B: 5x5
    # Split the inner dimension (5) into 2+3
    # A = [A_left | A_right] where A_left is 4x2, A_right is 4x3
    # B = [B_top; B_bot] where B_top is 2x5, B_bot is 3x5
    # C = A_left * B_top + A_right * B_bot
    
    # Now we need decompositions for:
    # 4x2 * 2x5 -> 4x5 (rank at most 4*2*5 = 40, but can be less)
    # 4x3 * 3x5 -> 4x5 (rank at most 4*3*5 = 60, but can be less)
    
    # For 4x2 * 2x5:
    # Split 4 into 2+2, 5 into 2+2+1
    # A_left (4x2) = [A_lt; A_lb] (each 2x2)
    # B_top (2x5) = [B_t1 | B_t2 | B_t3] (2x2, 2x2, 2x1)
    # A_left * B_top = A_lt*B_t1 + A_lt*B_t2 + A_lt*B_t3 + A_lb*B_t1 + A_lb*B_t2 + A_lb*B_t3
    # Each 2x2*2x2 uses rank 7 (Strassen)
    # Each 2x2*2x1 uses rank 4 (schoolbook, since one side is 2x1)
    # Actually 2x2 * 2x1: rank is at most 2*2*1 = 4, and it's tight.
    # So: 6 terms of rank 7? No, 4 terms of 2x2*2x2 (rank 7 each) + 2 terms of 2x2*2x1 (rank 4 each)
    # Wait: A_lt*B_t1, A_lt*B_t2 are 2x2*2x2 (rank 7 each)
    # A_lt*B_t3, A_lb*B_t3 are 2x2*2x1 (rank 4 each)
    # A_lb*B_t1, A_lb*B_t2 are 2x2*2x2 (rank 7 each)
    # Total for 4x2*2x5: 3*7 + 3*4 = 21 + 12 = 33
    
    # For 4x3 * 3x5:
    # Split 4 into 2+2, 3 into 1+2, 5 into 2+2+1
    # A_right (4x3) = [A_rt | A_rr] where A_rt is 4x1, A_rr is 4x2
    # Split A_rt into 2+2 rows: A_rt_top (2x1), A_rt_bot (2x1)
    # A_rr into 2+2 rows: A_rr_top (2x2), A_rr_bot (2x2)
    # B_bot (3x5) = [B_b1; B_b2; B_b3] where B_b1 is 1x5, B_b2 is 2x5, B_b3 is 0x5? No.
    # B_bot is 3x5, split rows into 1+2: B_b_top (1x5), B_b_bot (2x5)
    # B_b_top (1x5) split cols into 2+2+1: B_bt1 (1x2), B_bt2 (1x2), B_bt3 (1x1)
    # B_b_bot (2x5) split cols into 2+2+1: B_bb1 (2x2), B_bb2 (2x2), B_bb3 (2x1)
    
    # A_right * B_bot = A_rt * B_b_top + A_rr * B_b_bot
    # A_rt * B_b_top: (2x1 + 2x1) * (1x2 + 1x2 + 1x1)
    #   = A_rt_top*B_bt1 + A_rt_top*B_bt2 + A_rt_top*B_bt3 + A_rt_bot*B_bt1 + A_rt_bot*B_bt2 + A_rt_bot*B_bt3
    #   Each 2x1*1xk has rank at most 2*1*k = 2k
    #   2x1*1x2: rank 4, 2x1*1x1: rank 2
    #   Total: 4+4+2+4+4+2 = 20
    # A_rr * B_b_bot: (2x2 + 2x2) * (2x2 + 2x2 + 2x1)
    #   = A_rr_top*B_bb1 + A_rr_top*B_bb2 + A_rr_top*B_bb3 + A_rr_bot*B_bb1 + A_rr_bot*B_bb2 + A_rr_bot*B_bb3
    #   2x2*2x2: rank 7, 2x2*2x1: rank 4
    #   Total: 7+7+4+7+7+4 = 36
    # Total for 4x3*3x5: 20 + 36 = 56
    
    # Grand total: 33 + 56 = 89
    
    # Let me implement this more carefully by building the actual factor matrices.
    
    # I'll use a helper that accumulates rank-1 tensors.
    U = []  # will be list of (row, val) per term
    V = []
    W = []
    
    def add_tensor(u_entries, v_entries, w_entries):
        """Add a term. u_entries: list of (row_idx, val), v_entries: list of (row_idx, val), w_entries: list of (row_idx, val)
        where row_idx is in the global U, V, W row space."""
        U.append(u_entries)
        V.append(v_entries)
        W.append(w_entries)
    
    # Global dimensions:
    # U has n*m = 4*5 = 20 rows, indexed by i*m+j where i in [0,4), j in [0,5)
    # V has m*p = 5*5 = 25 rows, indexed by j*p+k where j in [0,5), k in [0,5)
    # W has p*n = 5*4 = 20 rows, indexed by k*n+i where k in [0,5), i in [0,4)
    
    # A is 4x5: A[i,j] for i in [0,4), j in [0,5)
    # B is 5x5: B[j,k] for j in [0,5), k in [0,5)
    # C is 4x5: C[i,k] for i in [0,4), k in [0,5)
    
    # Split inner dim j into [0,2) and [2,5)
    # A_left: rows i in [0,4), cols j in [0,2)
    # A_right: rows i in [0,4), cols j in [2,5)
    # B_top: rows j in [0,2), cols k in [0,5)
    # B_bot: rows j in [2,5), cols k in [0,5)
    
    # --- A_left (4x2) * B_top (2x5) ---
    # Split A_left rows into [0,2) and [2,4)
    # Split B_top cols into [0,2), [2,4), [4,5)
    # A_lt: 2x2 (i in [0,2), j in [0,2))
    # A_lb: 2x2 (i in [2,4), j in [0,2))
    # B_t1: 2x2 (j in [0,2), k in [0,2))
    # B_t2: 2x2 (j in [0,2), k in [2,4))
    # B_t3: 2x1 (j in [0,2), k in [4,5))
    
    # For each pair, use Strassen or schoolbook
    
    def strassen_2x2_2x2(i0, j0, k0, i1, j1, k1):
        """Add Strassen terms for A[i0:i1, j0:j1] * B[j0:j1, k0:k1] -> C[i0:i1, k0:k1]
        where all blocks are 2x2."""
        # Map local indices to global
        # A local (a, b) -> global U row: (i0+a)*m + (j0+b)
        # B local (b, c) -> global V row: (j0+b)*p + (k0+c)
        # C local (a, c) -> global W row: (k0+c)*n + (i0+a)
        
        local_tensors = _strassen_2x2()
        for u_local, v_local, w_local in local_tensors:
            u_entries = []
            v_entries = []
            w_entries = []
            for (ai, aj, val) in u_local:
                u_entries.append(((i0+ai)*m + (j0+aj), val))
            for (bj, bk, val) in v_local:
                v_entries.append(((j0+bj)*p + (k0+bk), val))
            for (ck, ci, val) in w_local:
                w_entries.append(((k0+ck)*n + (i0+ci), val))
            add_tensor(u_entries, v_entries, w_entries)
    
    def schoolbook_2x2_2x1(i0, j0, k0, i1, j1, k1):
        """Add schoolbook terms for A[i0:i1, j0:j1] * B[j0:j1, k0:k1] -> C[i0:i1, k0:k1]
        A is 2x2, B is 2x1."""
        for a in range(2):
            for b in range(2):
                for c in range(1):
                    add_tensor(
                        [((i0+a)*m + (j0+b), 1)],
                        [((j0+b)*p + (k0+c), 1)],
                        [((k0+c)*n + (i0+a), 1)]
                    )
    
    # A_lt * B_t1: 2x2 * 2x2, i in [0,2), j in [0,2), k in [0,2)
    strassen_2x2_2x2(0, 0, 0, 2, 2, 2)
    # A_lt * B_t2: 2x2 * 2x2, i in [0,2), j in [0,2), k in [2,4)
    strassen_2x2_2x2(0, 0, 2, 2, 2, 4)
    # A_lt * B_t3: 2x2 * 2x1, i in [0,2), j in [0,2), k in [4,5)
    schoolbook_2x2_2x1(0, 0, 4, 2, 2, 5)
    # A_lb * B_t1: 2x2 * 2x2, i in [2,4), j in [0,2), k in [0,2)
    strassen_2x2_2x2(2, 0, 0, 4, 2, 2)
    # A_lb * B_t2: 2x2 * 2x2, i in [2,4), j in [0,2), k in [2,4)
    strassen_2x2_2x2(2, 0, 2, 4, 2, 4)
    # A_lb * B_t3: 2x2 * 2x1, i in [2,4), j in [0,2), k in [4,5)
    schoolbook_2x2_2x1(2, 0, 4, 4, 2, 5)
    
    # --- A_right (4x3) * B_bot (3x5) ---
    # Split A_right cols into [2,3) and [3,5)
    # A_rc1: 4x1 (i in [0,4), j in [2,3))
    # A_rc2: 4x2 (i in [0,4), j in [3,5))
    # B_bot rows: j in [2,3) and [3,5)
    # B_b1: 1x5 (j in [2,3), k in [0,5))
    # B_b2: 2x5 (j in [3,5), k in [0,5))
    
    # A_rc1 * B_b1: 4x1 * 1x5
    # Split rows of A_rc1 into 2+2: i in [0,2) and [2,4)
    # Split cols of B_b1 into 2+2+1: k in [0,2), [2,4), [4,5)
    # 2x1 * 1x2: rank 4, 2x1 * 1x1: rank 2
    def schoolbook_2x1_1x2(i0, j0, k0, i1, j1, k1):
        """A[i0:i1, j0:j1] * B[j0:j1, k0:k1] where A is 2x1, B is 1x2"""
        for a in range(2):
            for b in range(1):
                for c in range(2):
                    add_tensor(
                        [((i0+a)*m + (j0+b), 1)],
                        [((j0+b)*p + (k0+c), 1)],
                        [((k0+c)*n + (i0+a), 1)]
                    )
    
    def schoolbook_2x1_1x1(i0, j0, k0, i1, j1, k1):
        """A[i0:i1, j0:j1] * B[j0:j1, k0:k1] where A is 2x1, B is 1x1"""
        for a in range(2):
            for b in range(1):
                for c in range(1):
                    add_tensor(
                        [((i0+a)*m + (j0+b), 1)],
                        [((j0+b)*p + (k0+c), 1)],
                        [((k0+c)*n + (i0+a), 1)]
                    )
    
    # A_rc1 top (i in [0,2)) * B_b1 col1 (k in [0,2)): 2x1 * 1x2
    schoolbook_2x1_1x2(0, 2, 0, 2, 3, 2)
    # A_rc1 top * B_b1 col2 (k in [2,4)): 2x1 * 1x2
    schoolbook_2x1_1x2(0, 2, 2, 2, 3, 4)
    # A_rc1 top * B_b1 col3 (k in [4,5)): 2x1 * 1x1
    schoolbook_2x1_1x1(0, 2, 4, 2, 3, 5)
    # A_rc1 bot (i in [2,4)) * B_b1 col1: 2x1 * 1x2
    schoolbook_2x1_1x2(2, 2, 0, 4, 3, 2)
    # A_rc1 bot * B_b1 col2: 2x1 * 1x2
    schoolbook_2x1_1x2(2, 2, 2, 4, 3, 4)
    # A_rc1 bot * B_b1 col3: 2x1 * 1x1
    schoolbook_2x1_1x1(2, 2, 4, 4, 3, 5)
    
    # A_rc2 (4x2) * B_b2 (2x5)
    # Split A_rc2 rows into 2+2: i in [0,2) and [2,4)
    # Split B_b2 cols into 2+2+1: k in [0,2), [2,4), [4,5)
    # A_rc2_top (2x2, i in [0,2), j in [3,5)) * B_b2_col1 (2x2, j in [3,5), k in [0,2)): 2x2*2x2
    strassen_2x2_2x2(0, 3, 0, 2, 5, 2)
    # A_rc2_top * B_b2_col2 (2x2, k in [2,4)): 2x2*2x2
    strassen_2x2_2x2(0, 3, 2, 2, 5, 4)
    # A_rc2_top * B_b2_col3 (2x1, k in [4,5)): 2x2*2x1
    schoolbook_2x2_2x1(0, 3, 4, 2, 5, 5)
    # A_rc2_bot (2x2, i in [2,4), j in [3,5)) * B_b2_col1: 2x2*2x2
    strassen_2x2_2x2(2, 3, 0, 4, 5, 2)
    # A_rc2_bot * B_b2_col2: 2x2*2x2
    strassen_2x2_2x2(2, 3, 2, 4, 5, 4)
    # A_rc2_bot * B_b2_col3: 2x2*2x1
    schoolbook_2x2_2x1(2, 3, 4, 4, 5, 5)
    
    # Now convert to matrix form
    rank = len(U)
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (row, val) in U[col]:
            U_mat[row][col] = val
        for (row, val) in V[col]:
            V_mat[row][col] = val
        for (row, val) in W[col]:
            W_mat[row][col] = val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
