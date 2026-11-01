"""Matrix multiplication 2x3x7 - hybrid Strassen/Schoolbook decomposition, rank 39."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7

    U_cols = []  # Each element is a column (list of length n*m)
    V_cols = []  # Each element is a column (list of length m*p)
    W_cols = []  # Each element is a column (list of length p*n)

    # Block 1: A[0:2, 0] (2x1) * B[0, 0:7] (1x7) -> C[0:2, 0:7]
    # Schoolbook: 2*1*7 = 14 terms
    for i in range(n):
        for k in range(p):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[i * m + 0] = 1.0
            V_col[0 * p + k] = 1.0
            W_col[k * n + i] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # Block 2: A[0:2, 1:3] (2x2) * B[1:3, 0:7] (2x7) -> C[0:2, 1:7]
    # Split B into 3 blocks of width 2 and 1 block of width 1
    # Strassen for each 2x2x2, schoolbook for 2x2x1

    def add_strassen_2x2x2(a_col_offset, b_row_offset, b_col_offset):
        """
        Add Strassen decomposition for A[a_rows, a_col_offset:a_col_offset+2]
        * B[b_row_offset:b_row_offset+2, b_col_offset:b_col_offset+2]
        -> C[a_rows, b_col_offset:b_col_offset+2]
        
        a_col_offset: column offset in A (0-indexed), here should be 1
        b_row_offset: row offset in B, here should be 1
        b_col_offset: column offset in B, here 0, 2, or 4
        """
        # A indices: U[i * m + (a_col_offset + j)] for i in {0,1}, j in {0,1}
        # B indices: V[(b_row_offset + i) * p + (b_col_offset + j)] for i in {0,1}, j in {0,1}
        # C indices: W[(b_col_offset + j) * n + i] for i in {0,1}, j in {0,1}
        
        def a_idx(i, j):
            return i * m + (a_col_offset + j)
        
        def b_idx(i, j):
            return (b_row_offset + i) * p + (b_col_offset + j)
        
        def c_idx(i, j):
            return (b_col_offset + j) * n + i
        
        # 7 Strassen terms
        # M1 = (A00+A11)(B00+B11)
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[a_idx(0, 0)] = 1.0
        U_col[a_idx(1, 1)] = 1.0
        V_col[b_idx(0, 0)] = 1.0
        V_col[b_idx(1, 1)] = 1.0
        W_col[c_idx(0, 0)] = 1.0
        W_col[c_idx(1, 1)] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # M2 = (A10+A11)B00
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[a_idx(1, 0)] = 1.0
        U_col[a_idx(1, 1)] = 1.0
        V_col[b_idx(0, 0)] = 1.0
        W_col[c_idx(1, 0)] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # M3 = A00(B10-B11)
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[a_idx(0, 0)] = 1.0
        V_col[b_idx(1, 0)] = 1.0
        V_col[b_idx(1, 1)] = -1.0
        W_col[c_idx(0, 1)] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # M4 = A11(B01-B00)
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[a_idx(1, 1)] = 1.0
        V_col[b_idx(0, 1)] = 1.0
        V_col[b_idx(0, 0)] = -1.0
        W_col[c_idx(0, 0)] = 1.0
        W_col[c_idx(1, 0)] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # M5 = (A00+A10)B11
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[a_idx(0, 0)] = 1.0
        U_col[a_idx(1, 0)] = 1.0
        V_col[b_idx(1, 1)] = 1.0
        W_col[c_idx(0, 0)] = -1.0
        W_col[c_idx(0, 1)] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # M6 = (A11-A00)(B00+B10)
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[a_idx(1, 1)] = 1.0
        U_col[a_idx(0, 0)] = -1.0
        V_col[b_idx(0, 0)] = 1.0
        V_col[b_idx(1, 0)] = 1.0
        W_col[c_idx(1, 0)] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # M7 = (A01-A11)(B11+B01)
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[a_idx(0, 1)] = 1.0
        U_col[a_idx(1, 1)] = -1.0
        V_col[b_idx(1, 1)] = 1.0
        V_col[b_idx(0, 1)] = 1.0
        W_col[c_idx(0, 0)] = -1.0
        W_col[c_idx(1, 1)] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    def add_schoolbook_2x2x1(a_col_offset, b_row_offset, b_col_offset):
        """
        Schoolbook for A[0:2, a_col_offset:a_col_offset+2] * B[b_row_offset:b_row_offset+2, b_col_offset]
        -> C[0:2, b_col_offset]
        """
        for i in range(2):
            for j in range(2):
                U_col = [0.0] * (n * m)
                V_col = [0.0] * (m * p)
                W_col = [0.0] * (p * n)
                U_col[i * m + (a_col_offset + j)] = 1.0
                V_col[(b_row_offset + i) * p + b_col_offset] = 1.0
                W_col[b_col_offset * n + i] = 1.0
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)

    # Apply Strassen to 3 blocks: cols 1-2, 3-4, 5-6
    add_strassen_2x2x2(a_col_offset=1, b_row_offset=1, b_col_offset=1)
    add_strassen_2x2x2(a_col_offset=1, b_row_offset=1, b_col_offset=3)
    add_strassen_2x2x2(a_col_offset=1, b_row_offset=1, b_col_offset=5)
    
    # Schoolbook for last column: col 7 (index 6)
    add_schoolbook_2x2x1(a_col_offset=1, b_row_offset=1, b_col_offset=7)

    # Build final matrices
    rank = len(U_cols)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for c in range(rank):
        for r in range(n * m):
            U[r][c] = U_cols[c][r]
        for r in range(m * p):
            V[r][c] = V_cols[c][r]
        for r in range(p * n):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
