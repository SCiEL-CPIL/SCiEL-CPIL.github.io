import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 3, 4, 6

    def ui(i, j): return i * m + j
    def vi(j, k): return j * p + k
    def wi(k, i): return k * n + i

    # Strassen 2x2x2 decomposition (7 terms)
    s_defs = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]

    # We use a hybrid approach:
    # - Row 0 of A: schoolbook (24 mults)
    # - Rows 1-2 of A: Strassen-style block decomposition
    #   Split A[1:3] into 2x2 blocks horizontally (cols 0-1, cols 2-3)
    #   Split B into 3 vertical blocks of width 2
    #   For each of the 3 B-blocks and 2 A-blocks: 7 mults each
    #   Total: 3 * 2 * 7 = 42 mults
    #   Grand total: 24 + 42 = 66

    rank = 66
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0

    def add_term(u, v, w):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u:
            U[idx, col] += val
        for idx, val in v:
            V[idx, col] += val
        for idx, val in w:
            W[idx, col] += val
        col += 1

    # Part 1: Schoolbook for row 0
    for j in range(m):
        for k in range(p):
            u = [(ui(0, j), 1)]
            v = [(vi(j, k), 1)]
            w = [(wi(k, 0), 1)]
            add_term(u, v, w)

    # Part 2: Strassen for rows 1-2
    for b_block in range(p // 2):
        b_col_start = b_block * 2

        # Left A-block (cols 0,1) * B rows 0,1
        for a_c, b_c, c_c in s_defs:
            u, v, w = [], [], []
            for idx, val in enumerate(a_c):
                if val != 0:
                    r, c = divmod(idx, 2)
                    u.append((ui(1 + r, c), val))
            for idx, val in enumerate(b_c):
                if val != 0:
                    r, c = divmod(idx, 2)
                    v.append((vi(r, b_col_start + c), val))
            for idx, val in enumerate(c_c):
                if val != 0:
                    r, c = divmod(idx, 2)
                    w.append((wi(b_col_start + c, 1 + r), val))
            add_term(u, v, w)

        # Right A-block (cols 2,3) * B rows 2,3
        for a_c, b_c, c_c in s_defs:
            u, v, w = [], [], []
            for idx, val in enumerate(a_c):
                if val != 0:
                    r, c = divmod(idx, 2)
                    u.append((ui(1 + r, 2 + c), val))
            for idx, val in enumerate(b_c):
                if val != 0:
                    r, c = divmod(idx, 2)
                    v.append((vi(2 + r, b_col_start + c), val))
            for idx, val in enumerate(c_c):
                if val != 0:
                    r, c = divmod(idx, 2)
                    w.append((wi(b_col_start + c, 1 + r), val))
            add_term(u, v, w)

    # Trim zero columns
    mask = np.any(U != 0, axis=0) & np.any(V != 0, axis=0) & np.any(W != 0, axis=0)
    U = U[:, mask]
    V = V[:, mask]
    W = W[:, mask]

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
