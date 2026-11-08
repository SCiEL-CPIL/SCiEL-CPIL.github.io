import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s

    n, m, p = 3, 4, 6

    # Indexing helpers
    def ui(i, j): return i * m + j
    def vi(j, k): return j * p + k
    def wi(k, i): return k * n + i

    # Precompute Strassen factors for 2x2x2
    s_defs = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]

    # We construct a decomposition of rank 66.
    # The decomposition is based on:
    # 1. Row 0 of A (1x4) multiplied by B (4x6) using schoolbook (24 mults).
    # 2. Rows 1-2 of A (2x4) multiplied by B (4x6) using Strassen for 2x2 blocks.
    #    B is split into 3 blocks of width 2.
    #    A[1:3] is split into 2 blocks of width 2.
    #    For each B block k (0,1,2):
    #       We compute A[1:3, 0:2] * B[0:2, 2k:2k+2] (7 mults)
    #       and A[1:3, 2:4] * B[2:4, 2k:2k+2] (7 mults)
    #    Total for Strassen part: 3 * (7 + 7) = 42 mults.
    #    Total rank = 24 + 42 = 66.

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

    # Part 1: Schoolbook for Row 0 of A
    # A[0, j] * B[j, k] -> C[k, 0]
    for j in range(m):
        for k in range(p):
            u = [(ui(0, j), 1)]
            v = [(vi(j, k), 1)]
            w = [(wi(k, 0), 1)]
            add_term(u, v, w)

    # Part 2: Strassen for Rows 1-2 of A
    # A_sub is 2x4. B is 4x6. C_sub is 2x6.
    # We iterate over B column blocks (size 2).
    for b_block in range(p // 2):
        b_col_start = b_block * 2
        
        # Left half of A_sub (cols 0,1) * Left half of B_block (rows 0,1)
        # This contributes to C_sub[:, b_col_start:b_col_start+2]
        # A_L is A[1:3, 0:2], B_L is B[0:2, b_col_start:b_col_start+2]
        # C_L is C[1:3, b_col_start:b_col_start+2]
        for a_c, b_c, c_c in s_defs:
            u = []
            v = []
            w = []
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

        # Right half of A_sub (cols 2,3) * Right half of B_block (rows 2,3)
        # This contributes to C_sub[:, b_col_start:b_col_start+2]
        # A_R is A[1:3, 2:4], B_R is B[2:4, b_col_start:b_col_start+2]
        # C_R is C[1:3, b_col_start:b_col_start+2]
        for a_c, b_c, c_c in s_defs:
            u = []
            v = []
            w = []
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

    # Check if we ran out of time (unlikely for this fixed construction)
    if time.time() > deadline:
        pass

    # Trim zero columns if any (should be none for this construction)
    mask = np.any(U != 0, axis=0) & np.any(V != 0, axis=0) & np.any(W != 0, axis=0)
    U = U[:, mask]
    V = V[:, mask]
    W = W[:, mask]

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
