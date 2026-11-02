"""Matrix multiplication 2x2x4 via two independent Strassen blocks (rank 14)."""

def construct():
    n, m, p = 2, 2, 4
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    rank = 14

    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]

    # Strassen 7-term decomposition for 2x2x2.
    # Each term: (u_entries, v_entries, w_entries)
    # u_entries: list of (i, j, val) -> index i*m+j in U
    # v_entries: list of (j, k, val) -> index j*2+k in V (local 2x2)
    # w_entries: list of (k, i, val) -> index k*2+i in W (local 2x2)
    #
    # Standard Strassen for C = A*B where A is 2x2, B is 2x2, C is 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    terms = [
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(1,0,1),(0,0,-1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    # Block 1: columns 0-6, uses B columns 0-1 (k=0,1)
    # Block 2: columns 7-13, uses B columns 2-3 (k=2,3)
    for r in range(7):
        u_entries, v_entries, w_entries = terms[r]

        # Block 1
        col1 = r
        for i, j, val in u_entries:
            u_idx = i * m + j
            U[u_idx][col1] += val
        for j, k, val in v_entries:
            v_idx = j * p + k
            V[v_idx][col1] += val
        for k, i, val in w_entries:
            w_idx = k * n + i
            W[w_idx][col1] += val

        # Block 2 (shift k by 2)
        col2 = 7 + r
        for i, j, val in u_entries:
            u_idx = i * m + j
            U[u_idx][col2] += val
        for j, k, val in v_entries:
            v_idx = j * p + (k + 2)
            V[v_idx][col2] += val
        for k, i, val in w_entries:
            w_idx = (k + 2) * n + i
            W[w_idx][col2] += val

    return {"U": U, "V": V, "W": W}
