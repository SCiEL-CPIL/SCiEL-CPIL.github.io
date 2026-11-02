"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, rank 7)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Strassen, rank 7)
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] (Schoolbook, rank 4)
    # Block 4: A[0:2, 2:3] * B[2:3, 0:5] -> C[0:2, 0:5] (Schoolbook, rank 10)
    # Total Rank = 7 + 7 + 4 + 10 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_col):
        # a_rows, a_cols: range of i, j in A
        # b_rows, b_cols: range of j, k in B
        # c_rows, c_cols: range of k, i in C (note W index is k*n + i)
        # Strassen factors for 2x2 * 2x2
        # A indices: (i, j) -> U[i*m + j]
        # B indices: (j, k) -> V[j*p + k]
        # C indices: (k, i) -> W[k*n + i]
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00: a_rows[0], a_cols[0]
        # A11: a_rows[1], a_cols[1]
        # B00: b_rows[0], b_cols[0]
        # B11: b_rows[1], b_cols[1]
        
        # Helper to get global indices
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        # M1
        U[u_idx(a_rows[0], a_cols[0])][start_col] += 1
        U[u_idx(a_rows[1], a_cols[1])][start_col] += 1
        V[v_idx(b_rows[0], b_cols[0])][start_col] += 1
        V[v_idx(b_rows[1], b_cols[1])][start_col] += 1
        W[w_idx(c_rows[0], c_rows[0])][start_col] += 1 # C00
        W[w_idx(c_rows[0], c_rows[1])][start_col] += 1 # C01? No, C01 is row 0 col 1.
        # Wait, C is 2x2. c_rows are k indices (0,1), c_cols are i indices (0,1).
        # W index is k*n + i.
        # C00 (k=0, i=0): w_idx(0,0)
        # C01 (k=0, i=1): w_idx(0,1)
        # C10 (k=1, i=0): w_idx(1,0)
        # C11 (k=1, i=1): w_idx(1,1)
        
        # Let's map Strassen outputs to W indices
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add contributions to W for each M.
        # M1 contributes to C00 (+1) and C11 (+1)
        W[w_idx(c_rows[0], c_cols[0])][start_col] += 1
        W[w_idx(c_rows[1], c_cols[1])][start_col] += 1
        col_ret = start_col + 1
        
        # M2 = (A10 + A11)B00
        U[u_idx(a_rows[1], a_cols[0])][col_ret] += 1
        U[u_idx(a_rows[1], a_cols[1])][col_ret] += 1
        V[v_idx(b_rows[0], b_cols[0])][col_ret] += 1
        # C10 = M2 + M4 => M2 contributes +1 to C10
        # C11 = M1 - M2 + M3 + M6 => M2 contributes -1 to C11
        W[w_idx(c_rows[1], c_cols[0])][col_ret] += 1
        W[w_idx(c_rows[1], c_cols[1])][col_ret] -= 1
        col_ret += 1
        
        # M3 = A00(B10 - B11)
        U[u_idx(a_rows[0], a_cols[0])][col_ret] += 1
        V[v_idx(b_rows[1], b_cols[0])][col_ret] += 1
        V[v_idx(b_rows[1], b_cols[1])][col_ret] -= 1
        # C01 = M3 + M5 => M3 contributes +1 to C01
        # C11 = M1 - M2 + M3 + M6 => M3 contributes +1 to C11
        W[w_idx(c_rows[0], c_cols[1])][col_ret] += 1
        W[w_idx(c_rows[1], c_cols[1])][col_ret] += 1
        col_ret += 1
        
        # M4 = A11(B01 - B00)
        U[u_idx(a_rows[1], a_cols[1])][col_ret] += 1
        V[v_idx(b_rows[0], b_cols[1])][col_ret] += 1
        V[v_idx(b_rows[0], b_cols[0])][col_ret] -= 1
        # C00 = M1 + M4 - M5 + M7 => M4 contributes +1 to C00
        # C10 = M2 + M4 => M4 contributes +1 to C10
        W[w_idx(c_rows[0], c_cols[0])][col_ret] += 1
        W[w_idx(c_rows[1], c_cols[0])][col_ret] += 1
        col_ret += 1
        
        # M5 = (A00 + A01)B11
        U[u_idx(a_rows[0], a_cols[0])][col_ret] += 1
        U[u_idx(a_rows[0], a_cols[1])][col_ret] += 1
        V[v_idx(b_rows[1], b_cols[1])][col_ret] += 1
        # C00 = M1 + M4 - M5 + M7 => M5 contributes -1 to C00
        # C01 = M3 + M5 => M5 contributes +1 to C01
        W[w_idx(c_rows[0], c_cols[0])][col_ret] -= 1
        W[w_idx(c_rows[0], c_cols[1])][col_ret] += 1
        col_ret += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[u_idx(a_rows[1], a_cols[0])][col_ret] += 1
        U[u_idx(a_rows[0], a_cols[0])][col_ret] -= 1
        V[v_idx(b_rows[0], b_cols[0])][col_ret] += 1
        V[v_idx(b_rows[0], b_cols[1])][col_ret] += 1
        # C11 = M1 - M2 + M3 + M6 => M6 contributes +1 to C11
        W[w_idx(c_rows[1], c_cols[1])][col_ret] += 1
        col_ret += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[u_idx(a_rows[0], a_cols[1])][col_ret] += 1
        U[u_idx(a_rows[1], a_cols[1])][col_ret] -= 1
        V[v_idx(b_rows[1], b_cols[0])][col_ret] += 1
        V[v_idx(b_rows[1], b_cols[1])][col_ret] += 1
        # C00 = M1 + M4 - M5 + M7 => M7 contributes +1 to C00
        W[w_idx(c_rows[0], c_cols[0])][col_ret] += 1
        col_ret += 1
        
        return col_ret

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # a_rows=[0,1], a_cols=[0,1]
    # b_rows=[0,1], b_cols=[0,1]
    # c_rows=[0,1] (k), c_cols=[0,1] (i)
    col = add_strassen([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], col)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    # a_rows=[0,1], a_cols=[0,1]
    # b_rows=[0,1], b_cols=[2,3]
    # c_rows=[2,3] (k), c_cols=[0,1] (i)
    col = add_strassen([0,1], [0,1], [0,1], [2,3], [2,3], [0,1], col)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5]
    # Schoolbook rank 4
    # Terms:
    # A00 B04 -> C04 (k=4, i=0)
    # A01 B14 -> C04 (k=4, i=0)
    # A10 B04 -> C14 (k=4, i=1)
    # A11 B14 -> C14 (k=4, i=1)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Term 1: A00 B04
    U[u_idx(0, 0)][col] = 1
    V[v_idx(0, 4)][col] = 1
    W[w_idx(4, 0)][col] = 1
    col += 1
    
    # Term 2: A01 B14
    U[u_idx(0, 1)][col] = 1
    V[v_idx(1, 4)][col] = 1
    W[w_idx(4, 0)][col] = 1
    col += 1
    
    # Term 3: A10 B04
    U[u_idx(1, 0)][col] = 1
    V[v_idx(0, 4)][col] = 1
    W[w_idx(4, 1)][col] = 1
    col += 1
    
    # Term 4: A11 B14
    U[u_idx(1, 1)][col] = 1
    V[v_idx(1, 4)][col] = 1
    W[w_idx(4, 1)][col] = 1
    col += 1
    
    # Block 4: A[0:2, 2:3] * B[2:3, 0:5]
    # Schoolbook rank 10
    # A indices: i in [0,1], j=2
    # B indices: j=2, k in [0,1,2,3,4]
    # C indices: k in [0,1,2,3,4], i in [0,1]
    
    for i in range(2):
        for k in range(5):
            U[u_idx(i, 2)][col] = 1
            V[v_idx(2, k)][col] = 1
            W[w_idx(k, i)][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
