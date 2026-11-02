# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 150.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _build_universe(q, n):
    size = q ** n
    uni = np.empty((size, n), dtype=np.int8)
    for i in range(size):
        v = i
        for j in range(n - 1, -1, -1):
            uni[i, j] = v % q
            v //= q
    return uni

def _ball_indices(uni, q, n, radius):
    size = len(uni)
    idx = np.arange(size)
    parts = [idx]
    for pos in range(n):
        col = uni[:, pos]
        pv = q ** pos
        for d in range(1, q):
            new = (col + d) % q
            parts.append(idx - col * pv + new * pv)
    return np.concatenate(parts, axis=0)

def _greedy_cover(uni, ball, rng):
    size = len(uni)
    uncovered = np.ones(size, dtype=bool)
    chosen = []
    while uncovered.any():
        newly = uncovered[ball]
        counts = newly.sum(axis=1)
        best = int(np.argmax(counts + rng.random(size) * 1e-6))
        if counts[best] == 0:
            break
        chosen.append(best)
        uncovered[ball[best]] = False
    return chosen

def _removal_pass(chosen, ball):
    size = len(ball)
    chosen = list(chosen)
    improved = True
    while improved:
        improved = False
        cov = np.zeros(size, dtype=np.int32)
        for c in chosen:
            cov[ball[c]] += 1
        to_remove = [c for c in chosen if cov[ball[c]].min() >= 2]
        if to_remove:
            for c in to_remove:
                chosen.remove(c)
            improved = True
    return chosen

def _swap_local_search(chosen, ball, budget):
    chosen = list(chosen)
    size = len(ball)
    ball_sets = [set(row.tolist()) for row in ball]
    t0 = time.time()
    improved = True
    while improved and (time.time() - t0) < budget:
        improved = False
        cov = np.zeros(size, dtype=np.int32)
        for c in chosen:
            cov[ball[c]] += 1
        chosen_set = set(chosen)
        for c in list(chosen):
            if (time.time() - t0) >= budget:
                break
            U = ball[c][cov[ball[c]] == 1]
            if len(U) == 0:
                continue
            Uset = set(U.tolist())
            for w in range(size):
                if w in chosen_set:
                    continue
                if Uset <= ball_sets[w]:
                    new_chosen = [x for x in chosen if x != c] + [w]
                    new_chosen = _removal_pass(new_chosen, ball)
                    if len(new_chosen) < len(chosen):
                        chosen = new_chosen
                        chosen_set = set(chosen)
                        improved = True
                        break
    return chosen

def construct(q, n, radius):
    global _START
    _START = time.time()
    uni = _build_universe(q, n)
    ball = _ball_indices(uni, q, n, radius)
    size = len(uni)

    rng = np.random.default_rng(0)
    best = _removal_pass(_greedy_cover(uni, ball, rng), ball)
    best_len = len(best)

    seed = 1
    while _remaining() > 6.0:
        rng = np.random.default_rng(seed)
        cand = _removal_pass(_greedy_cover(uni, ball, rng), ball)
        if len(cand) < best_len:
            best = cand
            best_len = len(cand)
            if best_len <= 186:
                break
        seed += 1

    if _remaining() > 10.0:
        best = _swap_local_search(best, ball, _remaining() - 10.0)
        best = _removal_pass(best, ball)

    seen = set()
    out = []
    for c in best:
        w = tuple(int(x) for x in uni[c])
        if w not in seen:
            seen.add(w)
            out.append(list(w))
    return out
# EVOLVE-BLOCK-END
