"""Edge coloring with no rainbow 4-cycle. Rank-1 multiplicative design."""
import time
import random
import numpy as np

N = 16
M = N * (N - 1) // 2  # 120
_EDGE_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
_ALL_QUADS = [(a, b, c, d) for a in range(N) for b in range(a + 1, N)
              for c in range(b + 1, N) for d in range(c + 1, N)]


def _make_T(vals, K):
    T = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(i + 1, N):
            T[i, j] = T[j, i] = (vals[i] * vals[j]) % K
    return T


def _bad4_count(T):
    cnt = 0
    for (a, b, c, d) in _ALL_QUADS:
        s = np.array([T[a, b], T[b, c], T[c, d], T[d, a],
                      T[a, b], T[b, d], T[d, c], T[c, a],
                      T[a, c], T[c, b], T[b, d], T[d, a]], dtype=np.int64)
        if np.unique(s).size > 3:
            cnt += 1
    return cnt


def _distinct_used_set(T):
    s = set()
    for (i, j) in _EDGE_PAIRS:
        s.add(int(T[i, j]))
    return len(s)


def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before the 90s kill

    value_sets = [
        list(range(1, 11)),
        list(range(2, 12)),
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23],
        [2, 4, 8, 16, 3, 9, 27, 5, 25, 7],
        [1, 2, 4, 8, 3, 6, 12, 5, 10, 20],
        [1, 3, 9, 27, 81, 2, 6, 18, 54, 162],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ]
    K_choices = [120, 100, 90, 80, 72, 60, 50, 48, 40, 36]

    best_distinct = 0
    best_T = None

    def time_ok():
        return (time.time() - t0) < DEADLINE

    def try_assignment(vals, K):
        nonlocal best_distinct, best_T
        T = _make_T(vals, K)
        if _bad4_count(T) == 0:
            d = _distinct_used_set(T)
            if d > best_distinct:
                best_distinct = d
                best_T = T.copy()

    rng = random.Random(12345)

    # --- Phase 1: randomized search over value assignments ---
    for vs_idx, vs in enumerate(value_sets):
        if not time_ok():
            break
        K = K_choices[vs_idx % len(K_choices)]
        for _ in range(400):
            if not time_ok():
                break
            vals = [vs[rng.randrange(len(vs))] for _ in range(N)]
            try_assignment(vals, K)

    # --- Phase 2: repair search (flip vertex values to reduce bad4) ---
    for vs_idx, vs in enumerate(value_sets):
        if not time_ok():
            break
        K = K_choices[(vs_idx + 3) % len(K_choices)]
        for _ in range(600):
            if not time_ok():
                break
            cur = [vs[rng.randrange(len(vs))] for _ in range(N)]
            T = _make_T(cur, K)
            bad = _bad4_count(T)
            if bad == 0:
                d = _distinct_used_set(T)
                if d > best_distinct:
                    best_distinct = d
                    best_T = T.copy()
                continue
            for _ in range(200):
                if not time_ok():
                    break
                v = rng.randrange(N)
                cur[v] = vs[rng.randrange(len(vs))]
                T2 = _make_T(cur, K)
                b2 = _bad4_count(T2)
                if b2 <= bad:
                    bad = b2
                    T = T2
                    if bad == 0:
                        d = _distinct_used_set(T)
                        if d > best_distinct:
                            best_distinct = d
                            best_T = T.copy()
                        break

    # --- Phase 3: polish with alternate K on fresh assignments ---
    for vs in value_sets[:3]:
        if not time_ok():
            break
        for K in K_choices:
            if not time_ok():
                break
            for _ in range(200):
                if not time_ok():
                    break
                vals = [vs[rng.randrange(len(vs))] for _ in range(N)]
                try_assignment(vals, K)

    if best_T is None:
        return [0] * M
    return [int(best_T[i, j]) for (i, j) in _EDGE_PAIRS]
