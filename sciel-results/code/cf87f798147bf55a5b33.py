"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 135  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # We decompose:
    # A (2x4) = [A1, A2] where A1, A2 are 2x2
    # B (4x5) = [B1; B2] where B1, B2 are 2x5
    # C (2x5) = A1*B1 + A2*B2
    #
    # Further split each B_i (2x5) into [Bi1, Bi2, Bi3] where Bi1,Bi2 are 2x2, Bi3 is 2x1
    # C = A1*B11 + A1*B12 + A1*B13 + A2*B21 + A2*B22 + A2*B23
    #
    # 2x2 * 2x2: Strassen (7 mults) -> 4 blocks * 7 = 28
    # 2x2 * 2x1: Trivial (4 mults) -> 2 blocks * 4 = 8
    # Total rank: 36
    
    U = []  # rows: n*m = 8
    V = []  # rows: m*p = 20
    W = []  # rows: p*n = 10
    
    rank = 0
    
    def add_factor(u_row_vals, v_row_vals, w_row_vals):
        """Add one rank-1 factor. u_row_vals: list of (row, val), etc."""
        nonlocal rank
        U.append([0] * rank)
        V.append([0] * rank)
        W.append([0] * rank)
        # Actually, let's build as lists and transpose at end
        rank += 1
        # Store the column index implicitly
        return rank - 1
    
    # Better approach: build U, V, W as lists of columns, then transpose
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        """u_entries: list of (row_idx, value), etc."""
        col_U = [0] * (n * m)  # 8
        col_V = [0] * (m * p)  # 20
        col_W = [0] * (p * n)  # 10
        for r, val in u_entries:
            col_U[r] = val
        for r, val in v_entries:
            col_V[r] = val
        for r, val in w_entries:
            col_W[r] = val
        U_cols.append(col_U)
        V_cols.append(col_V)
        W_cols.append(col_W)
    
    # Helper: A[i,j] is at row i*m+j = i*4+j in U
    # Helper: B[j,k] is at row j*p+k = j*5+k in V
    # Helper: C[i,k] is at row k*n+i = k*2+i in W (W[k*n+i, r])
    
    # Strassen for 2x2 * 2x2:
    # A_sub = [[a00, a01], [a10, a11]]
    # B_sub = [[b00, b01], [b10, b11]]
    # where a_ij is A[ai + i, aj + j], b_ij is B[bj + i, bk + j]
    # C_sub = [[c00, c01], [c10, c11]]
    # c_ij is C[ci + i, ck + j]
    
    def strassen_2x2(ai, aj, bj, bk, ci, ck):
        """
        ai, aj: top-left of A subblock (2x2)
        bj, bk: top-left of B subblock (2x2)
        ci, ck: top-left of C subblock (2x2)
        """
        a00 = (ai, aj)
        a01 = (ai, aj + 1)
        a10 = (ai + 1, aj)
        a11 = (ai + 1, aj + 1)
        
        b00 = (bj, bk)
        b01 = (bj, bk + 1)
        b10 = (bj + 1, bk)
        b11 = (bj + 1, bk + 1)
        
        c00 = (ci, ck)
        c01 = (ci, ck + 1)
        c10 = (ci + 1, ck)
        c11 = (ci + 1, ck + 1)
        
        # M1 = (a00+a11)*(b00+b11)
        add_term(
            [a00, a11],  # U: rows for a00 and a11
            [b00, b11],  # V: rows for b00 and b11
            [c00, c11]   # W: rows for c00 and c11
        )
        # Wait, this isn't right. Each term is a single rank-1 contribution.
        # M1 = (A00+A11)*(B00+B11) contributes to C00 and C11
        # But it's ONE multiplication, so it's one rank-1 term in the tensor.
        # The rank-1 term has U picking out (A00+A11), V picking out (B00+B11), 
        # and W distributing to C00 and C11.
        
        # Actually for tensor decomposition, each "multiplication" M is a rank-1 term:
        # U picks a linear combination of A entries, V picks a linear combo of B entries,
        # W picks a linear combo of C entries (with the product going to those C positions).
        # 
        # For M1 = (A00+A11)*(B00+B11):
        #   U has 1 at rows for A00 and A11
        #   V has 1 at rows for B00 and B11
        #   W has 1 at rows for C00 and C11
        # This is ONE rank-1 term.
        
        # Let me redo this properly.
        pass
    
    # Let me restart with a cleaner approach.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_rows, v_rows, w_rows):
        """
        u_rows: list of (row_index_in_U, coefficient)
        v_rows: list of (row_index_in_V, coefficient)
        w_rows: list of (row_index_in_W, coefficient)
        """
        col_U = [0] * (n * m)
        col_V = [0] * (m * p)
        col_W = [0] * (p * n)
        for r, val in u_rows:
            col_U[r] += val
        for r, val in v_rows:
            col_V[r] += val
        for r, val in w_rows:
            col_W[r] += val
        U_cols.append(col_U)
        V_cols.append(col_V)
        W_cols.append(col_W)
    
    def A_row(i, j):
        return i * m + j
    
    def B_row(j, k):
        return j * p + k
    
    def C_row(i, k):
        return k * n + i  # W[k*n+i, r]
    
    def strassen_block(ai, aj, bj, bk, ci, ck):
        """
        Strassen for A[ai:ai+2, aj:aj+2] * B[bj:bj+2, bk:bk+2] -> C[ci:ci+2, ck:ck+2]
        """
        a00 = A_row(ai, aj)
        a01 = A_row(ai, aj + 1)
        a10 = A_row(ai + 1, aj)
        a11 = A_row(ai + 1, aj + 1)
        
        b00 = B_row(bj, bk)
        b01 = B_row(bj, bk + 1)
        b10 = B_row(bj + 1, bk)
        b11 = B_row(bj + 1, bk + 1)
        
        c00 = C_row(ci, ck)
        c01 = C_row(ci, ck + 1)
        c10 = C_row(ci + 1, ck)
        c11 = C_row(ci + 1, ck + 1)
        
        # M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
        add_term([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, 1)])
        # M2 = (A10+A11)*B00 -> contributes to C10
        add_term([(a10, 1), (a11, 1)], [(b00, 1)], [(c10, 1)])
        # M3 = A00*(B10-B00) -> contributes to C01
        add_term([(a00, 1)], [(b10, 1), (b00, -1)], [(c01, 1)])
        # M4 = A11*(B11-B01) -> contributes to C00, C10
        add_term([(a11, 1)], [(b11, 1), (b01, -1)], [(c00, 1), (c10, 1)])
        # M5 = (A00+A01)*B11 -> contributes to C01, C11
        add_term([(a00, 1), (a01, 1)], [(b11, 1)], [(c01, 1), (c11, 1)])
        # M6 = (A10-A00)*(B00+B01) -> contributes to C11
        add_term([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c11, 1)])
        # M7 = (A01-A11)*(B10+B11) -> contributes to C00, C11
        add_term([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c00, 1), (c11, 1)])
    
    def trivial_2x2_2x1(ai, aj, bj, bk, ci, ck):
        """
        Trivial for A[ai:ai+2, aj:aj+2] * B[bj:bj+2, bk:bk+1] -> C[ci:ci+2, ck:ck+1]
        """
        a00 = A_row(ai, aj)
        a01 = A_row(ai, aj + 1)
        a10 = A_row(ai + 1, aj)
        a11 = A_row(ai + 1, aj + 1)
        
        b0 = B_row(bj, bk)
        b1 = B_row(bj + 1, bk)
        
        c0 = C_row(ci, ck)
        c1 = C_row(ci + 1, ck)
        
        # C0 = A00*B0 + A01*B1
        add_term([(a00, 1)], [(b0, 1)], [(c0, 1)])
        add_term([(a01, 1)], [(b1, 1)], [(c0, 1)])
        # C1 = A10*B0 + A11*B1
        add_term([(a10, 1)], [(b0, 1)], [(c1, 1)])
        add_term([(a11, 1)], [(b1, 1)], [(c1, 1)])
    
    # Now build the decomposition:
    # A (2x4): A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B (4x5): B1 = B[0:2, :], B2 = B[2:4, :]
    # B1 (2x5): B11 = B1[:, 0:2], B12 = B1[:, 2:4], B13 = B1[:, 4:5]
    # B2 (2x5): B21 = B2[:, 0:2], B22 = B2[:, 2:4], B23 = B2[:, 4:5]
    #
    # C = A1*B11 + A1*B12 + A1*B13 + A2*B21 + A2*B22 + A2*B23
    #
    # A1*B11: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A1*B12: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A1*B13: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    # A2*B21: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A2*B22: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # A2*B23: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    
    # Strassen blocks (2x2 * 2x2):
    strassen_block(0, 0, 0, 0, 0, 0)   # A1*B11 -> C[0:2, 0:2]
    strassen_block(0, 0, 0, 2, 0, 2)   # A1*B12 -> C[0:2, 2:4]
    strassen_block(0, 2, 2, 0, 0, 0)   # A2*B21 -> C[0:2, 0:2]
    strassen_block(0, 2, 2, 2, 0, 2)   # A2*B22 -> C[0:2, 2:4]
    
    # Trivial blocks (2x2 * 2x1):
    trivial_2x2_2x1(0, 0, 0, 4, 0, 4)   # A1*B13 -> C[0:2, 4:5]
    trivial_2x2_2x1(0, 2, 2, 4, 0, 4)   # A2*B23 -> C[0:2, 4:5]
    
    # Total: 4*7 + 2*4 = 28 + 8 = 36
    
    # Transpose columns to rows
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    # Verify the decomposition
    if time.time() - start_time < DEADLINE:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # The decomposition says: for each i, j, k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is valid, 0 otherwise
        # Actually, the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # should equal 1 for all (i,j,k) since matrix multiplication is the identity tensor.
        
        # Let's verify by checking the reconstructed multiplication
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # In tensor form: T[i,j,k] = 1 for all i,j,k
        # The decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        
        # Check a few entries
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1) > max_err:
                        max_err = abs(val - 1)
        
        if max_err > 1e-10:
            # Fallback to schoolbook
            U_fb = [[0] * (n*m*p) for _ in range(n * m)]
            V_fb = [[0] * (n*m*p) for _ in range(m * p)]
            W_fb = [[0] * (n*m*p) for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j][col] = 1
                        V_fb[j*p+k][col] = 1
                        W_fb[k*n+i][col] = 1
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
