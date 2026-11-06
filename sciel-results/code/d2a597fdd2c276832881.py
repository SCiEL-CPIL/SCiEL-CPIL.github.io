"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 82.0   # safety margin under the 90 s kill


def _largest_sidon(arr):
    """Exact largest subset whose unordered pair sums (incl. diagonals) are unique."""
    n = len(arr)
    s = [2 * x for x in arr]
    for i in range(n):
        for j in range(i + 1, n):
            s.append(arr[i] + arr[j])
    vals = sorted(set(s))
    idx = {v: i for i, v in enumerate(vals)}
    m = len(vals)
    M = 1 << m
    mask = [0] * M
    for i in range(n):
        for j in range(i, n):
            mask[1 << idx[s[i * n + j]]] = (1 << i) | (1 << j)
    best = 0
    for sub in range(1, 1 << n):
        used = 0
        ok = True
        for i in range(n):
            if sub & (1 << i):
                for j in range(i, n):
                    if sub & (1 << j):
                        used |= mask[1 << idx[s[i * n + j]]]
                        if used & (1 << i):
                            ok = False
                            break
                if not ok:
                    break
        if ok:
            c = sub.bit_count()
            if c > best:
                best = c
    return best


def _candidate(arr):
    return _largest_sidon(arr)


def _local_search(arr, rng, budget):
    arr = list(arr)
    cur = _candidate(arr)
    best = cur
    best_arr = list(arr)
    t = 1.0
    tmin = 0.02
    n = len(arr)
    while time.time() < budget:
        i = rng.randrange(n)
        old = arr[i]
        arr[i] = rng.randrange(0, 129)
        if arr[i] in arr[:i] or arr[i] in arr[i + 1:]:
            arr[i] = old
            continue
        new = _candidate(arr)
        if new <= cur or rng.random() < pow(2.718281828459045, -(new - cur) / t):
            cur = new
            if new < best:
                best = new
                best_arr = list(arr)
        else:
            arr[i] = old
        t = max(tmin, t * 0.9995)
    return best, best_arr


def _seed_candidates(rng):
    cands = []
    # dense low cluster + spread anchors (many a+b=c+d and 2*anchor=a+b collisions)
    base = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    cands.append(list(base))
    # cluster + large anchors
    for a1 in range(20, 60, 5):
        for a2 in range(60, 110, 8):
            cands.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, a1, a2])
    # random
    for _ in range(40):
        cands.append(rng.sample(range(0, 129), 12))
    # structured: multiples + jitter
    for step in (5, 7, 9):
        cands.append([(i * step) % 128 for i in range(12)])
    return cands


def construct():
    rng = random.Random(20240517)
    best = 12
    best_arr = list(range(12))
    budget = _DEADLINE - 1.5   # leave margin for return
    for cand in _seed_candidates(rng):
        if time.time() > budget:
            break
        v = _candidate(cand)
        if v < best:
            best = v
            best_arr = list(cand)
    # refine best with local search
    while time.time() < budget:
        v, va = _local_search(best_arr, rng, budget)
        if v < best:
            best = v
            best_arr = va
        if best <= 5:   # 5 is a strong lower bound for 12 points; stop early
            break
    return sorted(best_arr)
# EVOLVE-BLOCK-END
