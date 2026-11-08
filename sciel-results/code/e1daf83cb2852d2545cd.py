import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120

    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    sU = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    sV = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    sW = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)]
    ]

    blocks = [
        (0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6),
        (0, 2, 0), (0, 2, 2), (0, 2, 4), (0, 2, 6)
    ]

    for b_idx, (i_off, j_off, k_off) in enumerate(blocks):
        base_rank = b_idx * 7
        for r in range(7):
            cr = base_rank + r
            for idx, val in sU[r]:
                li, lj = divmod(idx, 2)
                gi = li + i_off
                gj = lj + j_off
                U[gi * m + gj, cr] += val
            for idx, val in sV[r]:
                lj, lk = divmod(idx, 2)
                gj = lj + j_off
                gk = lk + k_off
                V[gj * p + gk, cr] += val
            for idx, val in sW[r]:
                lk, li = divmod(idx, 2)
                gk = lk + k_off
                gi = li + i_off
                W[gk * n + gi, cr] += val

    # Build target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    # Check for linear dependencies among rank-1 terms
    flat = np.zeros((n * m * m * p * p * n, rank))
    for r in range(rank):
        T_r = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        flat[:, r] = T_r.flatten()

    tol = 1e-10
    keep = list(range(rank))
    for r in range(rank - 1, -1, -1):
        if time.time() > deadline - 20:
            break
        if r not in keep:
            continue
        test_keep = [k for k in keep if k != r]
        if not test_keep:
            continue
        cols = flat[:, test_keep]
        try:
            x, res, rk, sv = np.linalg.lstsq(cols, flat[:, r], rcond=tol)
            norm_r = np.linalg.norm(flat[:, r])
            if norm_r > 0 and (len(res) == 0 or res[0] < tol * norm_r):
                keep = test_keep
        except Exception:
            pass

    if len(keep) < rank:
        new_rank = len(keep)
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        T_new = np.zeros((n * m, m * p, p * n))
        for r in range(new_rank):
            T_new += np.einsum('a,b,c->abc', U_new[:, r], V_new[:, r], W_new[:, r])
        if np.max(np.abs(T_new - target)) < 1e-8:
            rank = new_rank
            U, V, W = U_new, V_new, W_new

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
