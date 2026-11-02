import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 120.0  # explicit guard, safety margin before 150s kill

    n, m, p = 2, 3, 9
    num_blocks = p // 3  # 3 blocks of width 3
    m3 = 3               # inner m per block
    n3 = 2               # inner n per block
    p3 = 3               # inner p per block

    def schoolbook_block():
        r = n3 * m3 * p3  # 18
        U = np.zeros((n3 * m3, r))
        V = np.zeros((m3 * p3, r))
        W = np.zeros((p3 * n3, r))
        c = 0
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    U[i * m3 + j, c] = 1.0
                    V[j * p3 + k, c] = 1.0
                    W[k * n3 + i, c] = 1.0
                    c += 1
        return U, V, W

    def build_block(r):
        # random rank-r candidate, verify, return (U,V,W) or None
        U = np.random.randn(n3 * m3, r)
        V = np.random.randn(m3 * p3, r)
        W = np.zeros((p3 * n3, r))
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    W[k * n3 + i, r - 1 - (i * m3 * p3 + j * p3 + k)] = 1.0
        # verify
        ok = True
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    s = float(np.dot(U[i * m3 + j, :], V[j * p3 + k, :]))
                    if abs(s - 1.0) > 1e-7:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            return None
        return U, V, W

    def try_find_block(r, budget):
        best = None
        t0 = time.time()
        while time.time() < budget:
            res = build_block(r)
            if res is not None:
                best = res
                break
        return best

    # try descending ranks, each with a time budget
    block = None
    remaining = deadline - time.time()
    for r in (6, 8, 10, 12, 14, 16):
        if remaining < 5:
            break
        budget = min(remaining - 3, 12.0)
        if budget > 2:
            block = try_find_block(r, time.time() + budget)
        if block is not None:
            break
        remaining = deadline - time.time()

    if block is None:
        # guaranteed fallback: schoolbook block (rank 18)
        block = schoolbook_block()
        block_rank = 18
    else:
        block_rank = block[0].shape[1]

    total_rank = block_rank * num_blocks
    U, Vb, Wb = block

    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))

    for b in range(num_blocks):
        col = b * block_rank
        # A rows: i in 0..1, j in 0..2  -> row index i*3+j
        U_full[:, col:col + block_rank] = Ub  # Ub shape (6, block_rank)
        # B columns for this block: j in 0..2, k in 0..2 -> col index j*9 + (b*3+k)
        for j in range(m3):
            for k in range(p3):
                V_full[j * p + (b * p3 + k), col:col + block_rank] = \
                    Vb[j * p3 + k, :]
        # W output: k in 0..2, i in 0..1 -> row index k*2 + i ; output col b*3+k
        for k in range(p3):
            for i in range(n3):
                W_full[(b * p3 + k) * n + i, col:col + block_rank] = \
                    Wb[k * n3 + i, :]

    # final safety check (cheap, 162 dots)
    if time.time() < deadline - 2:
        good = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(U_full[i * m + j, :], V_full[j * p + k, :]))
                    if abs(s - 1.0) > 1e-7:
                        good = False
                        break
                if not good:
                    break
            if not good:
                break
        if not good:
            # rebuild pure schoolbook (definitely valid)
            U_full, V_full, W_full = _full_schoolbook(n, m, p)

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}


def _full_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W
