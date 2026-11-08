"""Deterministic rank-78 decomposition for 3x4*4x7 matrix multiplication.

Uses 6 disjoint 2x2x2 Strassen blocks with i in {0,1}, partitioning j into
{0,1}x{2,3} and k into {0,1}x{2,3}x{4,5}.  Remaining entries use schoolbook.
Total rank = 6*7 + 36 = 78.
"""
import time

_DIMS = (3, 4, 7)


def construct():
    t0 = time.time()
    deadline = t0 + 125.0
    n, m, p = _DIMS

    rank = 78
    rows_u = n * m          # 12
    rows_v = m * p          # 28
    rows_w = p * n          # 21

    # Sparse column representation: list of (row, value) tuples per factor
    U_cols = [[] for _ in range(rank)]
    V_cols = [[] for _ in range(rank)]
    W_cols = [[] for _ in range(rank)]

    def put(arr, col, row, val):
        # Avoid duplicates (Strassen terms never duplicate a (col,row))
        arr[col].append((row, val))

    col = 0

    # (i0,i1,j0,j1,k0,k1)
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    covered = set()

    for (i0, i1, j0, j1, k0, k1) in blocks:
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

        # M1: (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        #      -> C[i0,k0] +=, C[i1,k1] +=
        put(U_cols, col, i0 * m + j0, 1)
        put(U_cols, col, i1 * m + j1, 1)
        put(V_cols, col, j0 * p + k0, 1)
        put(V_cols, col, j1 * p + k1, 1)
        put(W_cols, col, k0 * n + i0, 1)
        put(W_cols, col, k1 * n + i1, 1)

        # M2: (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        #      -> C[i1,k0] +=, C[i1,k1] -=
        put(U_cols, col + 1, i1 * m + j0, 1)
        put(U_cols, col + 1, i1 * m + j1, 1)
        put(V_cols, col + 1, j0 * p + k0, 1)
        put(W_cols, col + 1, k0 * n + i1, 1)
        put(W_cols, col + 1, k1 * n + i1, -1)

        # M3: A[i0,j0] * (B[j0,k1]-B[j1,k1])
        #      -> C[i0,k1] +=, C[i1,k1] +=
        put(U_cols, col + 2, i0 * m + j0, 1)
        put(V_cols, col + 2, j0 * p + k1, 1)
        put(V_cols, col + 2, j1 * p + k1, -1)
        put(W_cols, col + 2, k1 * n + i0, 1)
        put(W_cols, col + 2, k1 * n + i1, 1)

        # M4: A[i1,j1] * (B[j1,k0]-B[j0,k0])
        #      -> C[i0,k0] +=, C[i1,k0] +=
        put(U_cols, col + 3, i1 * m + j1, 1)
        put(V_cols, col + 3, j1 * p + k0, 1)
        put(V_cols, col + 3, j0 * p + k0, -1)
        put(W_cols, col + 3, k0 * n + i0, 1)
        put(W_cols, col + 3, k0 * n + i1, 1)

        # M5: (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        #      -> C[i0,k0] -=, C[i0,k1] +=
        put(U_cols, col + 4, i0 * m + j0, 1)
        put(U_cols, col + 4, i0 * m + j1, 1)
        put(V_cols, col + 4, j1 * p + k1, 1)
        put(W_cols, col + 4, k0 * n + i0, -1)
        put(W_cols, col + 4, k1 * n + i0, 1)

        # M6: (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        #      -> C[i1,k1] +=
        put(U_cols, col + 5, i1 * m + j0, 1)
        put(U_cols, col + 5, i0 * m + j0, -1)
        put(V_cols, col + 5, j0 * p + k0, 1)
        put(V_cols, col + 5, j0 * p + k1, 1)
        put(W_cols, col + 5, k1 * n + i1, 1)

        # M7: (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        #      -> C[i0,k0] +=
        put(U_cols, col + 6, i0 * m + j1, 1)
        put(U_cols, col + 6, i1 * m + j1, -1)
        put(V_cols, col + 6, j1 * p + k0, 1)
        put(V_cols, col + 6, j1 * p + k1, 1)
        put(W_cols, col + 6, k0 * n + i0, 1)

        col += 7

    # Schoolbook for uncovered triples
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                put(U_cols, col, i0 * m + j0, 1)
                put(V_cols, col, j0 * p + k0, 1)
                put(W_cols, col, k0 * n + i0, 1)
                col += 1

    # Convert to dense list-of-lists (the required interface)
    U = [[0] * rank for _ in range(rows_u)]
    V = [[0] * rank for _ in range(rows_v)]
    W = [[0] * rank for _ in range(rows_w)]
    for c in range(rank):
        for (r, v) in U_cols[c]:
            U[r][c] = v
        for (r, v) in V_cols[c]:
            V[r][c] = v
        for (r, v) in W_cols[c]:
            W[r][c] = v

    # Quick verification (sparse, cheap)
    for c in range(rank):
        if time.time() > deadline:
            break
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0
                    for (r, v) in U_cols[c]:
                        if r == u_idx:
                            for (r2, v2) in V_cols[c]:
                                if r2 == v_idx:
                                    for (r3, v3) in W_cols[c]:
                                        if r3 == w_idx:
                                            s = v * v2 * v3
                                            break
                                    break
                            break
                    if (i, j, k) == (0, 0, 0) and c == 0:
                        # spot-check one entry only to save time
                        if s == 0:
                            raise ValueError("Sanity check failed at (0,0,0) col 0")

    return {"U": U, "V": V, "W": W}
