"""Matrix multiplication 3x4x6 with Strassen-based block decomposition."""

def construct():
    n, m, p = 3, 4, 6
    rank = 66
    U = [[0.0] * rank for _ in range(n * m)]  # 12 x 66
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 66
    W = [[0.0] * rank for _ in range(p * n)]  # 18 x 66

    col = 0

    # Strassen factors for 2x2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # U rows: A11, A12, A21, A22 (local row*2+col)
    # V rows: B11, B12, B21, B22 (local row*2+col)
    # W rows: C11, C12, C21, C22 (local col*2+row) -> k*n+i
    #
    # Strassen decomposition entries (u_val, v_val, w_val) for each factor:
    # factor 1: U: A11+1, A12+0, A21+0, A22+1; V: B11+1, B12+0, B21+0, B22+1; W: C11+1, C12+0, C21+0, C22+1
    # factor 2: U: A11+0, A12+0, A21+1, A22+1; V: B11+1, B12+0, B21+0, B22+0; W: C11+0, C12+0, C21+1, C22+1
    # factor 3: U: A11+1, A12+0, A21+0, A22+0; V: B11+0, B12+1, B21+0, B22+(-1); W: C11+0, C12+1, C21+0, C22+0
    # factor 4: U: A11+0, A12+0, A21+0, A22+1; V: B11+(-1), B12+0, B21+1, B22+0; W: C11+0, C12+0, C21+0, C22+1
    # factor 5: U: A11+1, A12+1, A21+0, A22+0; V: B11+0, B12+0, B21+0, B22+1; W: C11+0, C12+0, C21+0, C22+1
    # factor 6: U: A11+(-1), A12+0, A21+1, A22+0; V: B11+1, B12+1, B21+0, B22+0; W: C11+1, C12+0, C21+0, C22+0
    # factor 7: U: A11+0, A12+1, A21+0, A22+(-1); V: B11+0, B12+0, B21+1, B22+1; W: C11+0, C12+1, C21+0, C22+0

    strassen_factors = [
        # (u_vals for [A11,A12,A21,A22], v_vals for [B11,B12,B21,B22], w_vals for [C11,C12,C21,C22])
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 0, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0]),
    ]

    # For 1x2x2: rank 4
    # U has 1 row (the single row of A), V has 2 rows (B11, B12 -> actually B rows), W has 2 rows (C cols)
    # A is 1x2: A[0,0], A[0,1]
    # B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
    # C is 1x2: C[0,0], C[0,1]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # Simple bilinear: 4 terms
    factors_1x2x2 = [
        (1, 1, 1, 0),   # A[0]*B[0,0] -> C[0]
        (1, 0, 0, 1),   # A[0]*B[0,1] -> C[1]
        (1, 0, 1, 0),   # A[1]*B[1,0] -> C[0]
        (1, 0, 0, 1),   # A[1]*B[1,1] -> C[1]
    ]
    # Actually let me be more careful.
    # U row index: 0 (only one row)
    # V row indices: 0, 1 (two columns of B, i.e., B[0,*] and B[1,*])
    # W row indices: 0, 1 (two columns of C)
    # Term 1: U[0]=1, V[0]=1, W[0]=1  -> A0 * B_row0_col0 * C_col0
    # Term 2: U[0]=1, V[1]=1, W[0]=1  -> A1 * B_row1_col0 * C_col0
    # Term 3: U[0]=1, V[0]=1, W[1]=1  -> A0 * B_row0_col1 * C_col1
    # Term 4: U[0]=1, V[1]=1, W[1]=1  -> A1 * B_row1_col1 * C_col1
    # So: (u_val, v_row, v_val, w_row, w_val)
    factors_1x2x2 = [
        (1, 0, 1, 0, 1),
        (1, 1, 1, 0, 1),
        (1, 0, 1, 1, 1),
        (1, 1, 1, 1, 1),
    ]

    # Helper to add a 2x2x2 Strassen block
    def add_strassen_222(base_i, base_j, base_k):
        """
        base_i: starting global row index for i (0 or 1)
        base_j: starting global column index for j (0 or 2)
        base_k: starting global column index for k (0, 2, or 4)
        """
        global col
        for (u_vals, v_vals, w_vals) in strassen_factors:
            # u_vals: [A11, A12, A21, A22]
            # A11 -> i=base_i+0, j=base_j+0
            # A12 -> i=base_i+0, j=base_j+1
            # A21 -> i=base_i+1, j=base_j+0
            # A22 -> i=base_i+1, j=base_j+1
            U[(base_i + 0) * m + (base_j + 0)][col] = u_vals[0]
            U[(base_i + 0) * m + (base_j + 1)][col] = u_vals[1]
            U[(base_i + 1) * m + (base_j + 0)][col] = u_vals[2]
            U[(base_i + 1) * m + (base_j + 1)][col] = u_vals[3]

            # v_vals: [B11, B12, B21, B22]
            # B11 -> j=base_j+0, k=base_k+0
            # B12 -> j=base_j+0, k=base_k+1
            # B21 -> j=base_j+1, k=base_k+0
            # B22 -> j=base_j+1, k=base_k+1
            V[(base_j + 0) * p + (base_k + 0)][col] = v_vals[0]
            V[(base_j + 0) * p + (base_k + 1)][col] = v_vals[1]
            V[(base_j + 1) * p + (base_k + 0)][col] = v_vals[2]
            V[(base_j + 1) * p + (base_k + 1)][col] = v_vals[3]

            # w_vals: [C11, C12, C21, C22]
            # C11 -> k=base_k+0, i=base_i+0
            # C12 -> k=base_k+0, i=base_i+1
            # C21 -> k=base_k+1, i=base_i+0
            # C22 -> k=base_k+1, i=base_i+1
            W[(base_k + 0) * n + (base_i + 0)][col] = w_vals[0]
            W[(base_k + 0) * n + (base_i + 1)][col] = w_vals[1]
            W[(base_k + 1) * n + (base_i + 0)][col] = w_vals[2]
            W[(base_k + 1) * n + (base_i + 1)][col] = w_vals[3]

            col += 1

    # Helper to add a 1x2x2 block
    def add_1x2x2(base_i, base_j, base_k):
        """
        base_i: global row index for i (only 0)
        base_j: starting global column index for j (0 or 2)
        base_k: starting global column index for k (0, 2, or 4)
        """
        global col
        for (u_val, v_row, v_val, w_row, w_val) in factors_1x2x2:
            # U: single row at i=base_i, j varies
            # A has columns j=base_j, base_j+1
            # U row index: base_i * m + (base_j + 0) for A0, base_i * m + (base_j + 1) for A1
            # Wait, U is indexed by i*m+j. For 1x2x2, there's only one i.
            # The two elements of A are A[0, base_j] and A[0, base_j+1]
            # U row for A[0, base_j+0]: base_i*m + base_j + 0
            # U row for A[0, base_j+1]: base_i*m + base_j + 1
            # But in our factorization, we have one U entry per term.
            # Let me re-think.
            # U matrix row = i*m + j. For 1x2x2, i is fixed, j has 2 values.
            # V matrix row = j*p + k. j has 2 values, k has 2 values.
            # W matrix row = k*n + i. k has 2 values, i is fixed.
            
            # Term: A[j_local] * B[j_local, k_local] -> C[k_local]
            # j_local is 0 or 1, k_local is 0 or 1
            
            # U row: base_i * m + base_j + 0 (always, since i is fixed and we multiply by the appropriate A element)
            # Wait no. The U matrix has entries for each (i,j) pair. In the decomposition,
            # each column c has U[i*m+j, c] values.
            # For term A[0,j0] * B[j0,k0] -> C[k0]:
            #   U[base_i*m + base_j + j0] gets u_val
            #   V[(base_j + j0)*p + base_k + k0] gets v_val
            #   W[(base_k + k0)*n + base_i] gets w_val
            
            # But in factors_1x2x2, I defined it as (u_val, v_row, v_val, w_row, w_val)
            # where v_row is the local j index and w_row is the local k index.
            # Let me redefine:
            pass

    # Let me redo 1x2x2 more carefully.
    # 4 terms:
    # Term 0: A[j=0] * B[j=0, k=0] -> C[k=0]
    #   U row: base_i*m + base_j + 0, value 1
    #   V row: (base_j+0)*p + base_k + 0, value 1
    #   W row: (base_k+0)*n + base_i, value 1
    # Term 1: A[j=1] * B[j=1, k=0] -> C[k=0]
    #   U row: base_i*m + base_j + 1, value 1
    #   V row: (base_j+1)*p + base_k + 0, value 1
    #   W row: (base_k+0)*n + base_i, value 1
    # Term 2: A[j=0] * B[j=0, k=1] -> C[k=1]
    #   U row: base_i*m + base_j + 0, value 1
    #   V row: (base_j+0)*p + base_k + 1, value 1
    #   W row: (base_k+1)*n + base_i, value 1
    # Term 3: A[j=1] * B[j=1, k=1] -> C[k=1]
    #   U row: base_i*m + base_j + 1, value 1
    #   V row: (base_j+1)*p + base_k + 1, value 1
    #   W row: (base_k+1)*n + base_i, value 1

    def add_1x2x2_v2(base_i, base_j, base_k):
        global col
        for (jl, kl) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            U[base_i * m + base_j + jl][col] = 1
            V[(base_j + jl) * p + base_k + kl][col] = 1
            W[(base_k + kl) * n + base_i][col] = 1
            col += 1

    # Now fill in the blocks:
    # Part 1: i=0, j in {0,1} and {2,3}, k in {0,1}, {2,3}, {4,5}
    # 2 j-blocks * 3 k-blocks = 6 blocks of 1x2x2, each rank 4 -> 24 columns
    for jb in [0, 2]:
        for kb in [0, 2, 4]:
            add_1x2x2_v2(0, jb, kb)

    # Part 2: i in {1,2}, j in {0,1} and {2,3}, k in {0,1}, {2,3}, {4,5}
    # 2 j-blocks * 3 k-blocks = 6 blocks of 2x2x2, each rank 7 -> 42 columns
    for jb in [0, 2]:
        for kb in [0, 2, 4]:
            add_strassen_222(1, jb, kb)

    return {"U": U, "V": V, "W": W}
