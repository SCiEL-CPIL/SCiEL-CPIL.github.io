"""Fresh design: optimize value set + edge selection for sum/product collisions."""
import time
import random
import numpy as np

N = 14
M = 35
MAXV = 10000
ALL_EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]  # 91 pairs
ALL_ARR = np.array(ALL_EDGES, dtype=np.int32)  # (91,2)
DEADLINE = 82.0  # wall-clock safety margin under 90s


def _score(vals, edges):
    v = np.asarray(vals, dtype=np.int64)
    a = v[edges[:, 0]]
    b = v[edges[:, 1]]
    s = len(np.unique(a + b))
    p = len(np.unique(a * b))
    return max(s, p)


def _score_edges(vals, edges):
    v = np.asarray(vals, dtype=np.int64)
    a = v[edges[:, 0]]
    b = v[edges[:, 1]]
    return max(len(np.unique(a + b)), len(np.unique(a * b)))


def _edge_pair_scores(vals):
    v = np.asarray(vals, dtype=np.int64)
    a = v[ALL_ARR[:, 0]]
    b = v[ALL_ARR[:, 1]]
    sums = a + b
    prods = a * b
    si = np.searchsorted(np.unique(sums), sums)
    pi = np.searchsorted(np.unique(prods), prods)
    return si, pi


def _greedy_edges(si, pi):
    """Pick M edges maximizing collisions (lowest combined counts)."""
    counts = np.zeros(N * N, dtype=np.int64)
    chosen = []
    for _ in range(M):
        cc = counts[si] + counts[pi]
        idx = int(np.argmin(cc))
        chosen.append(idx)
        counts[si[idx]] += 1
        counts[pi[idx]] += 1
    return np.array([ALL_ARR[i] for i in chosen], dtype=np.int32)


def _local_edges(si, pi, edges, iters=4000):
    """Swap local search to reduce max(distinct sums, distinct products)."""
    edges = edges.copy()
    chosen = np.zeros(len(ALL_EDGES), dtype=bool)
    chosen[edges[:, 0] * N + edges[:, 1]] = True

    def cur_score():
        v = None
        a = np.empty(M, dtype=np.int64)
        b = np.empty(M, dtype=np.int64)
        return None  # placeholder

    def score_of(e):
        a = si_e = None
        return None

    # recompute from vals via si/pi
    def sc(e):
        sidx = si[e[:, 0] * N + e[:, 1]]
        pidx = pi[e[:, 0] * N + e[:, 1]]
        return max(len(np.unique(sidx)), len(np.unique(pidx)))

    best = sc(edges)
    for _ in range(iters):
        i = random.randrange(M)
        cur = chosen[edges[i, 0] * N + edges[i, 1]]
        # find a non-chosen edge to swap in
        cand = np.flatnonzero(~chosen)
        if len(cand) == 0:
            break
        j = cand[random.randrange(len(cand))]
        new_edges = edges.copy()
        new_edges[i] = ALL_ARR[j]
        ns = sc(new_edges)
        if ns < best:
            best = ns
            edges = new_edges
            chosen[edges[i, 0] * N + edges[i, 1]] = False
            chosen[j] = True
    return edges, best


def _try_value_set(vals, deadline):
    vals = np.asarray(vals, dtype=np.int64)
    si, pi = _edge_pair_scores(vals)
    edges = _greedy_edges(si, pi)
    score = _score_edges(vals, edges)
    # local edge search
    edges, score = _local_edges(si, pi, edges, iters=3000)
    # value local search
    v = vals.copy()
    for _ in range(2000):
        if time.time() > deadline:
            break
        k = random.randrange(N)
        old = v[k]
        v[k] = random.randint(1, MAXV)
        while v[k] in v:
            v[k] = random.randint(1, MAXV)
        si2, pi2 = _edge_pair_scores(v)
        e2 = _greedy_edges(si2, pi2)
        s2 = _score_edges(v, e2)
        if s2 < score:
            score = s2
            si, pi = si2, pi2
            edges = e2
            vals = v.copy()
        else:
            v[k] = old
    return vals, edges, score


def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random(12345)
    best_score = 10 ** 9
    best_vals = None
    best_edges = None

    # initial value sets: arithmetic-like (many sum collisions), geometric-like (product collisions), random
    def arith(a, d):
        return np.array([a + d * k for k in range(N)], dtype=np.int64)

    candidates = [
        arith(1, 7),
        arith(2, 5),
        arith(3, 4),
        np.array([2 ** k for k in range(N)], dtype=np.int64) * 1,  # products collide? no, sums distinct
        np.array([3 ** k for k in range(N)], dtype=np.int64) % (MAXV + 1),
    ]
    for _ in range(40):
        c = np.array(sorted(rng.sample(range(1, MAXV + 1), N)), dtype=np.int64)
        candidates.append(c)

    for c in candidates:
        if time.time() > deadline:
            break
        if len(set(c.tolist())) != N or c.min() < 1 or c.max() > MAXV:
            continue
        vals, edges, sc = _try_value_set(c, deadline)
        if sc < best_score:
            best_score = sc
            best_vals = vals.copy()
            best_edges = edges.copy()

    # final value local search on best
    if best_vals is not None and time.time() < deadline - 2:
        best_vals, best_edges, best_score = _try_value_set(best_vals, deadline)

    if best_vals is None:
        # fallback
        best_vals = np.arange(1, N + 1, dtype=np.int64)
        best_edges = np.array(ALL_EDGES[:M], dtype=np.int32)

    return {
        'values': [int(x) for x in best_vals.tolist()],
        'edges': [[int(i), int(j)] for i, j in best_edges.tolist()],
    }
