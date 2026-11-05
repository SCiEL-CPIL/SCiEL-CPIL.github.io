# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def _time_budget(start, margin=8.0):
    return (time.time() - start) < (150.0 - margin)


def _fallback_tetracode(q, n, radius):
    # Valid for q=3, n>=4, radius>=1: direct sum of ternary tetracode.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def construct(q, n, radius):
    start = time.time()
    D = q ** n

    # Precompute space digits.
    space = np.empty((D, n), dtype=np.int64)
    for i in range(D):
        v = i
        for j in range(n - 1, -1, -1):
            space[i, j] = v % q
            v //= q

    base = [1] * n
    for j in range(n - 1, -1, -1):
        base[j] = q ** j

    covered = np.zeros(D, dtype=bool)
    code = []

    def neighbors_of_idx(idx):
        word = space[idx]
        if radius == 0:
            yield idx
            return
        for r2 in range(radius + 1):
            for combo in itertools.combinations(range(n), r2):
                alts = [(d for d in range(q) if d != word[j]) for j in combo]
                for vals in itertools.product(*alts):
                    y = idx
                    for j, nv in zip(combo, vals):
                        y += (nv - word[j]) * base[j]
                    yield y

    neigh_cache = {}

    def get_neighbors(idx):
        if idx not in neigh_cache:
            neigh_cache[idx] = list(neighbors_of_idx(idx))
        return neigh_cache[idx]

    def coverage_count(idx):
        cnt = 0
        for nb in get_neighbors(idx):
            if not covered[nb]:
                cnt += 1
        return cnt

    step = max(1, D // 20000)
    complete = False
    while True:
        if covered.all():
            complete = True
            break
        if not _time_budget(start):
            break
        best_idx = -1
        best_cnt = -1
        for idx in range(0, D, step):
            c = coverage_count(idx)
            if c > best_cnt:
                best_cnt = c
                best_idx = idx
                if best_cnt >= D:
                    break
        if best_idx == -1 or best_cnt <= 0:
            unc = np.flatnonzero(~covered)
            if unc.size == 0:
                complete = True
                break
            best_idx = int(unc[0])
        for nb in get_neighbors(best_idx):
            covered[nb] = True
        code.append(best_idx)

    if not complete:
        # Greedy didn't finish in time; return a guaranteed-valid construction.
        if q == 3 and n >= 4 and radius >= 1:
            return _fallback_tetracode(q, n, radius)
        # Generic valid fallback: every word (trivially valid).
        return [space[i].tolist() for i in range(D)]

    # Local search: remove redundant codewords.
    if _time_budget(start, margin=12.0):
        improved = True
        while improved and _time_budget(start, margin=12.0):
            improved = False
            m = len(code)
            for i in range(m):
                if not _time_budget(start, margin=12.0):
                    break
                cov = np.zeros(D, dtype=bool)
                for k in range(m):
                    if k == i:
                        continue
                    for nb in get_neighbors(code[k]):
                        cov[nb] = True
                if cov.all():
                    code.pop(i)
                    improved = True
                    break

    result = [space[idx].tolist() for idx in code]
    return result
# EVOLVE-BLOCK-END
