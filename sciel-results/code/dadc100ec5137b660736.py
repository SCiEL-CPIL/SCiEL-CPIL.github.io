import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False

        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)

        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # M2 = (c+d)e -> C10 += 1, C11 -= 1
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1

        # M3 = a(f-h) -> C01 += 1, C11 += 1
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # M4 = d(g-e) -> C00 += 1, C10 += 1
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

        # M5 = (a+b)h -> C00 -= 1, C01 += 1
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1

        # M6 = (c-a)(e+f) -> C11 += 1
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # M7 = (b-d)(g+h) -> C00 += 1
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

        return True

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False

        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0

        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        # P1 = a*e -> C00
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

        # P2 = b*g -> C00
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

        # P3 = c*e -> C10
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

        # P4 = d*g -> C10
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

        return True

    # Process blocks: split inner dimension j into {0,1} and {2,3}
    # For each, split output k into 2+2+2+1
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            if not add_strassen_block(0, j_start, k_start):
                break
        add_schoolbook_221_block(0, j_start, 6)

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify correctness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {s}")

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
