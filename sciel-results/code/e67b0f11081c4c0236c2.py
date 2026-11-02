"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Hybrid Strassen/schoolbook decomposition of rank 44 for 3x4 * 4x4.
    # U shape (12, 44), V shape (16, 44), W shape (12, 44).
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44

    # Block indices:
    # A rows: i=0,1 (top), i=2 (bottom)
    # A cols: j=0,1 (left), j=2,3 (right)
    # B rows: j=0,1 (left), j=2,3 (right)
    # B cols: k=0,1 (left), k=2,3 (right)
    # C rows: i=0,1 (top), i=2 (bottom)
    # C cols: k=0,1 (left), k=2,3 (right)

    col = 0

    def add_strassen(a_i0, a_i1, a_j0, a_j1, b_j0, b_j1, b_k0, b_k1, c_i0, c_i1, c_k0, c_k1):
        """
        Add a 2x2 Strassen decomposition for:
        A[i0:i1, j0:j1] * B[j0:j1, k0:k1] -> C[i0:i1, k0:k1]
        where i1-i0=2, j1-j0=2, k1-k0=2.
        """
        nonlocal col
        # M1 = (A[i0,i1, j0,j1] + A[i0,i1, j0,j1+1]) * (B[j0,j1, k0,k1] + B[j0,j1, k0,k1+1])
        # Wait, let me be more careful.
        # For 2x2 matrices:
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = A*B = [[c00, c01], [c10, c11]]
        #
        # Strassen:
        # M1 = (a00 + a11) * (b00 + b11)
        # M2 = (a10 + a11) * b00
        # M3 = a00 * (b01 - b11)
        # M4 = a11 * (b10 - b00)
        # M5 = (a00 + a01) * b11
        # M6 = (a10 - a00) * (b00 + b01)
        # M7 = (a01 - a11) * (b10 + b11)
        #
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6

        # Map indices:
        # A local (0,0) -> global (a_i0, a_j0)
        # A local (0,1) -> global (a_i0, a_j0+1)
        # A local (1,0) -> global (a_i0+1, a_j0)
        # A local (1,1) -> global (a_i0+1, a_j0+1)
        #
        # B local (0,0) -> global (b_j0, b_k0)
        # B local (0,1) -> global (b_j0, b_k0+1)
        # B local (1,0) -> global (b_j0+1, b_k0)
        # B local (1,1) -> global (b_j0+1, b_k0+1)

        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        U[a_i0 * 4 + a_j0][col] = 1
        U[(a_i0 + 1) * 4 + (a_j0 + 1)][col] = 1
        V[b_j0 * 4 + b_k0][col] = 1
        V[(b_j0 + 1) * 4 + (b_k0 + 1)][col] = 1
        W[b_k0 * 3 + c_i0][col] = 1
        W[b_k0 * 3 + c_i0][col] = 1  # c00 gets +M1
        W[b_k0 * 3 + c_i0][col] = 1  # duplicate, need to track signs

        # This approach is getting messy with direct addition. Let me use a different strategy.
        # I'll build the U, V, W matrices directly using a list of (U_idx, V_idx, W_idx) tuples with coefficients.

    # Let me restart with a cleaner approach: build lists of (u_idx, v_idx, w_idx, coeff)
    # Then accumulate into U, V, W.

    # Clear previous
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        if coeff != 0:
            U[u_idx][col] = coeff
            V[v_idx][col] = coeff
            W[w_idx][col] = coeff
        col += 1

    def add_strassen_block(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        """
        Add 7 Strassen terms for 2x2 block.
        A rows a_i0, a_i0+1; cols a_j0, a_j0+1
        B rows b_j0, b_j0+1; cols b_k0, b_k0+1
        C rows c_i0, c_i0+1; cols c_k0, c_k0+1
        """
        nonlocal col
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # C[0,0] += M1, C[1,1] -= M1
        # Actually: c00 = M1 + M4 - M5 + M7, c11 = M1 - M2 + M3 + M6
        # So M1 contributes +1 to c00 and +1 to c11.

        # M1
        add_term(a_i0 * 4 + a_j0, b_j0 * 4 + b_k0, c_k0 * 3 + c_i0, 1)      # c00
        add_term(a_i0 * 4 + a_j0, b_j0 * 4 + b_k0, c_k0 * 3 + c_i0, 1)  # duplicate, bad

        # This is wrong because each term is one column with one U, one V, one W.
        # I need to think of this differently.

        # Let me use a different approach: for each of the 7 Strassen multiplications,
        # I define the U, V, W vectors for that column.

        # Actually, the standard way is:
        # Each column c has:
        #   U[:, c] = vector in R^{n*m}
        #   V[:, c] = vector in R^{m*p}
        #   W[:, c] = vector in R^{p*n}
        # And the decomposition is: sum_c outer(U[:,c], V[:,c]) = T, where T is the multiplication tensor.
        # But actually, the tensor T has entries T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k].
        # The rank-r decomposition means T = sum_{c=0}^{r-1} u_c v_c w_c (outer products of 3 vectors).
        # Where u_c has entries indexed by (i,j), v_c by (j,k), w_c by (k,i).
        # And w_c[k*n+i] is the W entry.

        # So for each column c, I need to specify:
        #   For each (i,j): U[i*m+j, c]
        #   For each (j,k): V[j*p+k, c]
        #   For each (k,i): W[k*n+i, c]

        # Let me redo this properly.
    pass

    # Let me just directly enumerate all terms.
    # I'll use a list of dicts for each column.

    cols = []

    def add_col(u_dict, v_dict, w_dict):
        cols.append((u_dict, v_dict, w_dict))

    # Strassen 2x2 block helper:
    # A indices: rows (a_i0, a_i0+1), cols (a_j0, a_j0+1)
    # B indices: rows (b_j0, b_j0+1), cols (b_k0, b_k0+1)
    # C indices: rows (c_i0, c_i0+1), cols (c_k0, c_k0+1)

    def add_strassen(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # u: (a_i0, a_j0):1, (a_i0+1, a_j0+1):1
        # v: (b_j0, b_k0):1, (b_j0+1, b_k0+1):1
        # w: (c_k0, c_i0):1, (c_k0+1, c_i0+1):1
        add_col(
            {(a_i0, a_j0): 1, (a_i0+1, a_j0+1): 1},
            {(b_j0, b_k0): 1, (b_j0+1, b_k0+1): 1},
            {(c_k0, c_i0): 1, (c_k0+1, c_i0+1): 1}
        )

        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # u: (a_i0+1, a_j0):1, (a_i0+1, a_j0+1):1
        # v: (b_j0, b_k0):1
        # w: (c_k0, c_i0+1):1  [c10 += M2]
        add_col(
            {(a_i0+1, a_j0): 1, (a_i0+1, a_j0+1): 1},
            {(b_j0, b_k0): 1},
            {(c_k0, c_i0+1): 1}
        )

        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # u: (a_i0, a_j0):1
        # v: (b_j0, b_k0+1):1, (b_j0+1, b_k0+1):-1
        # w: (c_k0+1, c_i0):1  [c01 += M3]
        add_col(
            {(a_i0, a_j0): 1},
            {(b_j0, b_k0+1): 1, (b_j0+1, b_k0+1): -1},
            {(c_k0+1, c_i0): 1}
        )

        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # u: (a_i0+1, a_j0+1):1
        # v: (b_j0+1, b_k0):1, (b_j0, b_k0):-1
        # w: (c_k0, c_i0):1, (c_k0, c_i0+1):1  [c00 += M4, c10 += M4]
        add_col(
            {(a_i0+1, a_j0+1): 1},
            {(b_j0+1, b_k0): 1, (b_j0, b_k0): -1},
            {(c_k0, c_i0): 1, (c_k0, c_i0+1): 1}
        )

        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # u: (a_i0, a_j0):1, (a_i0, a_j0+1):1
        # v: (b_j0+1, b_k0+1):1
        # w: (c_k0, c_i0):-1, (c_k0+1, c_i0):1  [c00 -= M5, c01 += M5]
        add_col(
            {(a_i0, a_j0): 1, (a_i0, a_j0+1): 1},
            {(b_j0+1, b_k0+1): 1},
            {(c_k0, c_i0): -1, (c_k0+1, c_i0): 1}
        )

        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # u: (a_i0+1, a_j0):1, (a_i0, a_j0):-1
        # v: (b_j0, b_k0):1, (b_j0, b_k0+1):1
        # w: (c_k0+1, c_i0+1):1  [c11 += M6]
        add_col(
            {(a_i0+1, a_j0): 1, (a_i0, a_j0): -1},
            {(b_j0, b_k0): 1, (b_j0, b_k0+1): 1},
            {(c_k0+1, c_i0+1): 1}
        )

        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # u: (a_i0, a_j0+1):1, (a_i0+1, a_j0+1):-1
        # v: (b_j0+1, b_k0):1, (b_j0+1, b_k0+1):1
        # w: (c_k0, c_i0):1  [c00 += M7]
        add_col(
            {(a_i0, a_j0+1): 1, (a_i0+1, a_j0+1): -1},
            {(b_j0+1, b_k0): 1, (b_j0+1, b_k0+1): 1},
            {(c_k0, c_i0): 1}
        )

    def add_simple_block(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0, a_rows=1, a_cols=1, b_rows=1, b_cols=1):
        """
        Add schoolbook terms for a block.
        """
        for di in range(a_rows):
            for dj in range(a_cols):
                for dk in range(b_cols):
                    a_idx = (a_i0 + di, a_j0 + dj)
                    b_idx = (b_j0 + dj, b_k0 + dk)
                    c_idx = (c_k0 + dk, c_i0 + di)
                    add_col({a_idx: 1}, {b_idx: 1}, {c_idx: 1})

    # Now build the 44-column decomposition.

    # Part 1: C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    # Term 1.1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen(a_i0=0, a_j0=0, b_j0=0, b_k0=0, c_i0=0, c_k0=0)  # 7 cols
    # Term 1.2: Strassen for A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen(a_i0=0, a_j0=2, b_j0=2, b_k0=0, c_i0=0, c_k0=0)  # 7 cols

    # Part 2: C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    # Term 2.1: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen(a_i0=0, a_j0=0, b_j0=0, b_k0=2, c_i0=0, c_k0=2)  # 7 cols
    # Term 2.2: Strassen for A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen(a_i0=0, a_j0=2, b_j0=2, b_k0=2, c_i0=0, c_k0=2)  # 7 cols

    # Part 3: C[2, 0:2] = A[2, 0:2]*B[0:2, 0:2] + A[2, 2:4]*B[2:4, 0:2]
    # Term 3.1: Schoolbook for A[2, 0:2] * B[0:2, 0:2] (4 terms)
    add_simple_block(a_i0=2, a_j0=0, b_j0=0, b_k0=0, c_i0=2, c_k0=0, a_rows=1, a_cols=2, b_rows=2, b_cols=2)
    # Wait, A[2, 0:2] means i=2, j=0,1. So 1 row, 2 cols.
    # B[0:2, 0:2] means j=0,1; k=0,1. So 2 rows, 2 cols.
    # The product gives C[2, 0:2].
    # For each j in {0,1} and k in {0,1}: one multiplication.
    # That's 4 terms.
    # But add_simple_block with a_rows=1, a_cols=2, b_rows=2, b_cols=2 would iterate:
    # di in range(1), dj in range(2), dk in range(2) -> 4 terms. Correct.

    # Term 3.2: Schoolbook for A[2, 2:4] * B[2:4, 0:2] (4 terms)
    add_simple_block(a_i0=2, a_j0=2, b_j0=2, b_k0=0, c_i0=2, c_k0=0, a_rows=1, a_cols=2, b_rows=2, b_cols=2)

    # Part 4: C[2, 2:4] = A[2, 0:2]*B[0:2, 2:4] + A[2, 2:4]*B[2:4, 2:4]
    # Term 4.1: Schoolbook for A[2, 0:2] * B[0:2, 2:4] (4 terms)
    add_simple_block(a_i0=2, a_j0=0, b_j0=0, b_k0=2, c_i0=2, c_k0=2, a_rows=1, a_cols=2, b_rows=2, b_cols=2)

    # Term 4.2: Schoolbook for A[2, 2:4] * B[2:4, 2:4] (4 terms)
    add_simple_block(a_i0=2, a_j0=2, b_j0=2, b_k0=2, c_i0=2, c_k0=2, a_rows=1, a_cols=2, b_rows=2, b_cols=2)

    # Total: 7+7+7+7+4+4+4+4 = 44. Correct.

    # Now build U, V, W from cols.
    rank = len(cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u_dict, v_dict, w_dict) in enumerate(cols):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
