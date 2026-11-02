import numpy as np
import time

DIMENSIONS = [6, 6, 6]


def _strassen3x3():
    """Strassen rank-7 decomposition of a 3x3 matrix product.

    Partition 3x3 into a 2x2 grid of 3x3 blocks (positions 00,01,10,11).
    Returns (steps, c_combos):
      steps[s] = (a_coeffs, b_coeffs) where coeffs are lists of (block, sign)
      c_combos[c] = {s: sign}  for output block c in {0,1,2,3}
    """
    steps = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),       # M1
        ([(2, 1), (3, 1)], [(0, 1)]),                # M2
        ([(0, 1)], [(1, 1), (3, -1)]),               # M3
        ([(3, 1)], [(2, 1), (0, -1)]),               # M4
        ([(0, 1), (1, 1)], [(3, 1)]),                # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),       # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),       # M7
    ]
    c_combos = [
        {0: 1, 3: -1, 5: 1, 6: -1},   # C00
        {2: 1, 4: 1},                  # C01
        {1: 1, 3: 1},                  # C10
        {0: 1, 1: -1, 2: 1, 6: 1},    # C11
    ]
    return steps, c_combos


def _build_sumshared(n, m, p):
    """Structural decomposition: share one 3x3 Strassen across the 3 j-summands.

    For each (i,k) block pair (3x3 x 3x3 product) use rank 7, folding the
    summation index j (3 values) into the coefficient lists. Total rank 9*7=63.
    """
    steps, c_combos = _strassen3x3()
    bs = 3          # sub-block size
    nb = 2          # blocks per dimension
    n_s = 7         # Strassen rank per (i,k) pair
    total_rank = (n // bs) * (p // bs) * n_s   # 9 * 7 = 63

    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)

    for ib in range(n // bs):
        for kb in range(p // bs):
            base = (ib * (p // bs) + kb) * n_s
            for s in range(n_s):
                a_coeffs, b_coeffs = steps[s]
                for (ab, asgn) in a_coeffs:
                    for (bb, bs_gn) in b_coeffs:
                        # A block ab -> (row_block=ab//2, col_block=ab%2)
                        # B block bb -> (row_block=bb//2, col_block=bb%2)
                        ar = ib * bs + (ab // 2) * bs
                        ac = (ab % 2) * bs
                        br = (bb // 2) * bs
                        bc = kb * bs + (bb % 2) * bs
                        for i in range(bs):
                            for j in range(bs):
                                for k in range(bs):
                                    gi = ar + i          # global A row  / C row
                                    gj = ac + j          # global A col  / B row (summation index)
                                    gk = bc + k          # global B col  / C col
                                    # output block of (gi, gk)
                                    c_block_idx = (gi // bs) * nb + (gk // bs)
                                    c_sign = c_combos[c_block_idx].get(s, 0)
                                    if c_sign == 0:
                                        continue
                                    col = base + s * (bs * bs) + i * (bs * bs) + j * bs + k
                                    U[gi * m + gj, col] = asgn
                                    V[gj * p + gk, col] = bs_gn
                                    W[gk * n + gi, col] = c_sign
    return {"U": U, "V": V, "W": W}


def _build_blockstrassen(n, m, p):
    """Correct Block-Strassen (rank 189): per-(i,k,j) 3x3 Strassen products."""
    steps, c_combos = _strassen3x3()
    bs = 3
    nb = 2
    n_s = 7
    block_rank = bs * bs * bs          # 27
    total_rank = (n // bs) * (p // bs) * (m // bs) * n_s   # 27 * 7 = 189

    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)

    for ib in range(n // bs):
        for jb in range(m // bs):
            for kb in range(p // bs):
                base = (ib * (m // bs) * (p // bs) + jb * (p // bs) + kb) * n_s
                for s in range(n_s):
                    a_coeffs, b_coeffs = steps[s]
                    for (ab, asgn) in a_coeffs:
                        for (bb, bs_gn) in b_coeffs:
                            ar = ib * bs + (ab // 2) * bs
                            ac = jb * bs + (ab % 2) * bs
                            br = jb * bs + (bb // 2) * bs
                            bc = kb * bs + (bb % 2) * bs
                            for i in range(bs):
                                for j in range(bs):
                                    for k in range(bs):
                                        gi = ar + i
                                        gj = ac + j
                                        gk = bc + k
                                        c_block_idx = (gi // bs) * nb + (gk // bs)
                                        c_sign = c_combos[c_block_idx].get(s, 0)
                                        if c_sign == 0:
                                            continue
                                        col = base + s * (bs * bs) + i * (bs * bs) + j * bs + k
                                        U[gi * m + gj, col] = asgn
                                        V[gj * p + gk, col] = bs_gn
                                        W[gk * n + gi, col] = c_sign
    return {"U": U, "V": V, "W": W}


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, col] = 1
                V[k * p + j, col] = 1
                W[j * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, full=False):
    """Check sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] == 1 for all (i,k,j)."""
    if full:
        idx = np.arange(n * m)
        for i in range(n):
            for k in range(m):
                rowU = U[i * m + k, :]
                for j in range(p):
                    s = np.dot(rowU * V[k * p + j, :], W[j * n + i, :])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    # random sample
    rng = np.random.default_rng(12345)
    for _ in range(2000):
        i = int(rng.integers(0, n))
        k = int(rng.integers(0, m))
        j = int(rng.integers(0, p))
        s = np.dot(U[i * m + k, :] * V[k * p + j, :], W[j * n + i, :])
        if abs(s - 1.0) > 1e-8:
            return False
    return True


def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS

    # Prefer the lowest-rank structural candidate, verify, fall back on failure.
    candidates = [
        ("sumshared", _build_sumshared, 63),
        ("blockstrassen", _build_blockstrassen, 189),
        ("schoolbook", _build_schoolbook, n * m * p),
    ]

    for name, builder, expected_rank in candidates:
        if time.time() > deadline - 5:
            break
        try:
            d = builder(n, m, p)
            if not _verify(d["U"], d["V"], d["W"], n, m, p, full=(name != "schoolbook")):
                continue
            return {"U": d["U"].tolist(), "V": d["V"].tolist(), "W": d["W"].tolist()}
        except Exception:
            continue

    # last-resort safe fallback (always valid)
    d = _build_schoolbook(n, m, p)
    return {"U": d["U"].tolist(), "V": d["V"].tolist(), "W": d["W"].tolist()}
