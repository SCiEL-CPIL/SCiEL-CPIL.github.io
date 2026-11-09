import numpy as np
import time

DIMENSIONS = (4, 4, 7)

def construct():
    start = time.time()
    deadline = 130.0

    n, m, p = DIMENSIONS  # 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28

    # Rank 49 for 4x4x4 via Strassen, rank 48 for 4x4x3 via schoolbook
    rank_444 = 49
    rank_443 = 48
    total_rank = rank_444 + rank_443

    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)

    # Strassen 2x2x2 tensor: 7 bilinear forms
    # (X linear form, Y linear form, Z linear form)
    # Each form is list of (row, col, coeff)
    strassen_2x2 = [
        ([((0,0,1),(1,1,1))], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],           [(1,0,1),(1,1,-1)]),
        ([(0,0,1)],           [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)],           [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)],   [(1,1,1)],           [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)],  [(0,0,1),(0,1,1)],   [(1,1,1)]),
        ([(0,1,1),(1,1,-1)],  [(1,0,1),(1,1,1)],   [(0,0,1)]),
    ]

    # Part 1: 4x4x4 via nested Strassen (49 scalar products)
    # Block structure: A = [[A00,A01],[A10,A11]], each 2x2
    for s_idx in range(7):
        X_def, Y_def, Z_def = strassen_2x2[s_idx]
        for w_idx in range(7):
            u_def, v_def, out_def = strassen_2x2[w_idx]
            r = s_idx * 7 + w_idx

            # U column: coefficients of A entries (A is 4x4, flat index i*m+j)
            U_col = np.zeros(nm)
            for (bx, by, cx) in X_def:
                for (ii, jj, cu) in u_def:
                    i = 2 * bx + ii
                    j = 2 * by + jj
                    a_idx = i * m + j
                    U_col[a_idx] += cx * cu
            U[:, r] = U_col

            # V column: coefficients of B entries (B is 4x4, flat index row*p+col)
            V_col = np.zeros(mp)
            for (bx, by, cy) in Y_def:
                for (ii, jj, cv) in v_def:
                    row_b = 2 * bx + ii
                    col_b = 2 * by + jj
                    b_idx = row_b * p + col_b
                    V_col[b_idx] += cy * cv
            V[:, r] = V_col

            # W column: coefficients of C entries (C is 4x4, flat index col*n+row)
            W_col = np.zeros(pn)
            for (dx, dy, cd) in Z_def:
                for (ii, jj, co) in out_def:
                    row_c = 2 * dx + ii
                    col_c = 2 * dy + jj
                    c_idx = col_c * n + row_c
                    W_col[c_idx] += cd * co
            W[:, r] = W_col

    # Part 2: 4x4x3 schoolbook for B columns 4,5,6
    offset = rank_444
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                a_idx = i * m + j
                b_idx = j * p + (4 + k)
                c_idx = (4 + k) * n + i

                U[a_idx, offset + col] = 1.0
                V[b_idx, offset + col] = 1.0
                W[c_idx, offset + col] = 1.0
                col += 1

    # Verify target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :] * V[b, :], W[c, :])
                if abs(val - 1.0) > 1e-9:
                    raise ValueError(f"Target entry ({a},{b},{c}) = {val}, expected 1.0")

    # Verify non-target entries
    target_mask = np.zeros((nm, mp, pn), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_mask[a, b, c] = True

    T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    non_target_mask = ~target_mask
    if np.any(non_target_mask):
        max_non_target = np.abs(T[non_target_mask]).max()
        if max_non_target > 1e-9:
            worst = np.unravel_index(np.argmax(np.abs(T[non_target_mask])), (nm, mp, pn))
            raise ValueError(f"Non-target entry {worst} = {T[worst]}, expected 0.0")

    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
