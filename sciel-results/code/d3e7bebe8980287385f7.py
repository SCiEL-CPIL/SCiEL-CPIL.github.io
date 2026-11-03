"""QAP solver: diversified starts + fast 2-opt local search with O(n) swap deltas."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _cost(p, a, b):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(p, i, j, a, b, R, C):
    # R[k] = sum_l a[k,l] * b[p[k], p[j]]   (k's contribution toward j)
    # C[k] = sum_l a[k,l] * b[p[i], p[l]]   (i's contribution toward k)
    pi, pj = p[i], p[j]
    ai, aj = a[i], a[j]
    bi, bj = b[pi], b[pj]
    # pairwise term between i and j
    pair = (a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj]) - (a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi])
    # cross terms with all k (including i,j handled by subtracting their R/C)
    # sum_k a[i,k](b[pj,pk]-b[pi,pk]) = ai @ (b[:,pj]-b[:,pi])
    t1 = ai @ (b[:, pj] - b[:, pi])
    # sum_k a[k,i](b[pk,pj]-b[pk,pi]) = (a[:,i]) @ (b[:,pj]-b[:,pi])
    t2 = a[:, i] @ (b[:, pj] - b[:, pi])
    # sum_k a[j,k](b[pi,pk]-b[pj,pk]) = aj @ (b[:,pi]-b[:,pj])
    t3 = aj @ (b[:, pi] - b[:, pj])
    # sum_k a[k,j](b[pk,pi]-b[pk,pj]) = (a[:,j]) @ (b[:,pi]-b[:,pj])
    t4 = a[:, j] @ (b[:, pi] - b[:, pj])
    return pair + t1 + t2 + t3 + t4


def _local_search(p, a, b, budget, rng):
    n = len(p)
    p = p.copy()
    end = time.time() + budget
    improved = True
    while improved:
        if time.time() >= end - 0.5:
            break
        improved = False
        order = rng.permutation(n)
        for ii in range(n):
            i = order[ii]
            if time.time() >= end - 0.5:
                break
            for jj in range(ii + 1, n):
                j = order[jj]
                d = _swap_delta(p, i, j, a, b, None, None)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    improved = True
    return p


def solve(a, b):
    start = time.time()
    n = len(a)
    deadline = start + 110.0
    rng = np.random.default_rng(12345)

    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)

    def make_start(kind, k):
        if kind == 0:
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p
        elif kind == 1:
            heavy = np.argsort(-a.sum(axis=0), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            q = np.empty(n, dtype=np.int64)
            q[heavy] = central
            return q
        else:
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            idx = rng.choice(n, size=max(1, n // (k + 2)), replace=False)
            p[idx] = rng.permutation(p[idx])
            return p

    best = None
    best_cost = None

    def try_solution(p, budget):
        nonlocal best, best_cost
        p = _local_search(p, a, b, budget, rng)
        c = _cost(p, a, b)
        if best_cost is None or c < best_cost:
            best = p
            best_cost = c
        return c

    # Phase 1: long search on rank start
    p0 = make_start(0, 0)
    try_solution(p0, deadline - time.time() - 4.0)

    # Phase 2: a few more starts if time remains
    remaining = deadline - time.time() - 2.0
    for kind, k in [(1, 0), (2, 1)]:
        if remaining < 5.0:
            break
        budget = min(remaining * 0.4, 15.0)
        try_solution(make_start(kind, k), budget)
        remaining = deadline - time.time() - 2.0

    if best is None:
        return make_start(0, 0).tolist()
    return best.tolist()
# EVOLVE-BLOCK-END
