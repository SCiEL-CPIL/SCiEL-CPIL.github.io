import time
import random
import numpy as np
import itertools


def _precompute(q, n):
    total = q ** n
    digits_of = [None] * total
    for i in range(total):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
    neighbors = [None] * total
    for i in range(total):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            ov = d[pos]
            for nv in range(q):
                if nv == ov:
                    continue
                idx = 0
                for j in range(n):
                    v = nv if j == pos else d[j]
                    idx = idx * q + v
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)
    return digits_of, neighbors


def _greedy_init(total, neighbors, digits_of, deadline, rseed):
    rnd = random.Random(rseed)
    cov = np.zeros(total, dtype=np.int32)
    code = []
    while True:
        t = time.time()
        if t > deadline:
            break
        uncov = np.flatnonzero(cov == 0)
        if uncov.size == 0:
            break
        cands = np.unique(neighbors[uncov])
        if cands.size > 400:
            cands = rnd.sample(list(cands), 400)
        scores = np.array([int(np.sum(cov[neighbors[c]] == 0)) for c in cands], dtype=np.int64)
        best = scores.max()
        top = cands[scores == best]
        c = int(rnd.choice(top.tolist()))
        code.append(c)
        cov[neighbors[c]] += 1
    return code, cov


def _solve_minimal(total, A, code, cov, deadline):
    """Return a minimal code (each word covered exactly once) of size <= len(code), or None."""
    cur = sorted(set(code))
    b = np.ones(total, dtype=np.int64)
    for _ in range(200):
        if time.time() > deadline:
            return None
        cur = sorted(set(cur))
        if len(cur) == 0:
            return None
        M = A[:, cur]
        s = M.sum(axis=1)
        if np.all(s == 1):
            return cur
        over = np.flatnonzero(s > 1)
        if over.size == 0:
            return None
        for u in over:
            row = np.flatnonzero(M[u] == 1)
            if row.size == 0:
                return None
            c = int(cur[int(row[0])])
            if c in cur:
                cur.remove(c)
        if len(cur) < len(set(code)):
            return cur
    return None


def _vns(total, A, b, cur, code_size, deadline, rseed):
    """Variable-neighborhood search over minimal solutions; return (best_code, best_size)."""
    rnd = random.Random(rseed)
    best = cur[:]
    best_size = len(cur)
    while time.time() < deadline:
        cur = best[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            k = 1
            while k <= 2 and time.time() < deadline:
                if len(cur) <= 1:
                    break
                for _ in range(400):
                    if time.time() > deadline:
                        break
                    i1, i2 = rnd.sample(range(len(cur)), 2)
                    c1, c2 = cur[i1], cur[i2]
                    c1n = neighbors_global[c1]
                    c2n = neighbors_global[c2]
                    shared = set(c1n) & set(c2n)
                    if len(shared) < 2:
                        continue
                    shared = list(shared)
                    a1 = rnd.choice(shared)
                    for _ in range(30):
                        a2 = rnd.choice(shared)
                        if a2 == a1:
                            continue
                        # find replacement for c1 covering a1, not c2
                        R1 = np.flatnonzero(A[a1])
                        R1 = [int(x) for x in R1 if x != c1 and x != c2]
                        if not R1:
                            continue
                        R2 = np.flatnonzero(A[a2])
                        R2 = [int(x) for x in R2 if x != c1 and x != c2]
                        if not R2:
                            continue
                        c1_new = rnd.choice(R1)
                        c2_new = rnd.choice(R2)
                        if c1_new == c2_new:
                            continue
                        M = A[:, [c1, c2, c1_new, c2_new]]
                        rhs = np.array([1, 1, 1, 1], dtype=np.int64)
                        sub = M[:, [0, 1]] @ np.array([1, 1]) + M[:, [2, 3]] @ np.array([-1, -1])
                        if np.all(sub == 0):
                            new_cur = cur[:]
                            new_cur[i1] = c1_new
                            new_cur[i2] = c2_new
                            cur = new_cur
                            improved = True
                            if len(cur) < best_size:
                                best = cur[:]
                                best_size = len(cur)
                            break
                        if _ == 29:
                            pass
                if improved:
                    break
                k += 1
        if time.time() > deadline:
            break
    return best, best_size


neighbors_global = None


def construct(q, n, radius):
    global neighbors_global
    start = time.time()
    deadline = start + 115.0

    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    total = q ** n
    digits_of, neighbors = _precompute(q, n)
    neighbors_global = neighbors

    # Boolean neighbor matrix A: A[word, code] = 1 if word within radius of code
    A = np.zeros((total, total), dtype=bool)
    for i in range(total):
        A[i, neighbors[i]] = True

    # Phase 1: greedy init
    code, cov = _greedy_init(total, neighbors, digits_of, deadline - 70.0, 7)
    # Make minimal (exact)
    min_code = _solve_minimal(total, A, code, cov, deadline - 65.0)
    if min_code is None:
        # fallback: peel to minimal
        cur = sorted(set(code))
        for _ in range(5000):
            if time.time() > deadline - 60.0:
                break
            s = A[:, cur].sum(axis=1)
            if np.all(s == 1):
                break
            over = np.flatnonzero(s > 1)
            if over.size == 0:
                break
            u = over[0]
            row = np.flatnonzero(A[u, cur])
            c = int(cur[int(row[0])])
            if c in cur:
                cur.remove(c)
        min_code = cur

    # Phase 2: VNS from minimal solution
    best, best_size = _vns(total, A, np.ones(total, dtype=np.int64), min_code, len(min_code),
                           deadline - 8.0, 42)

    # Final safety: verify best is a valid covering
    final_cov = A[:, best].sum(axis=1)
    if not np.all(final_cov >= 1):
        final_cov2 = A[:, min_code].sum(axis=1)
        if np.all(final_cov2 >= 1):
            best = min_code
        else:
            best = list(range(total))

    return [list(digits_of[i]) for i in best]
