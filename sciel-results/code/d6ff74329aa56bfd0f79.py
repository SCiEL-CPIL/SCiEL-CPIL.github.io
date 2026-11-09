import time
import numpy as np

DEADLINE = time.time() + 120.0
N = 1000

def _sum_counts(indicator):
    # number of pairs (a<=b) in the set that sum to each s
    return np.convolve(indicator, indicator)

def _valid(indicator):
    return bool(np.all(_sum_counts(indicator) <= 2))

def _greedy():
    counts = np.zeros(2 * N + 1, dtype=np.int64)
    chosen = []
    for x in range(1, N + 1):
        ok = True
        for y in chosen:
            s = x + y
            if counts[s] + 1 > 2:
                ok = False
                break
        if ok and counts[2 * x] + 1 > 2:
            ok = False
        if ok:
            chosen.append(x)
            for y in chosen:
                counts[x + y] += 1
    return chosen

def _refine(chosen):
    ind = np.zeros(N + 1, dtype=np.int8)
    for x in chosen:
        ind[x] = 1
    cur = set(chosen)
    improved = True
    while improved and time.time() < DEADLINE - 1.0:
        improved = False
        for x in list(cur):
            ind[x] = 0
            base = _sum_counts(ind)
            for y in range(1, N + 1):
                if y in cur:
                    continue
                if base[2 * y] >= 2:
                    continue
                if np.any(base[x + 1 : x + N + 1] >= 2):
                    continue
                cur.remove(x)
                cur.add(y)
                ind[x] = 0
                ind[y] = 1
                improved = True
                break
            if improved:
                ind[x] = 1  # restore x
                for z in cur:
                    ind[z] = 1
                ind[x] = 0
            else:
                ind[x] = 1
    return cur

def construct():
    t0 = time.time()
    # start from greedy
    cur = set(_greedy())
    # multi-refinement to escape local optima
    best = cur
    for _ in range(200):
        if time.time() > DEADLINE - 2.0:
            break
        cand = _refine(cur)
        if len(cand) > len(best):
            best = cand
        cur = cand
    return sorted(best)
