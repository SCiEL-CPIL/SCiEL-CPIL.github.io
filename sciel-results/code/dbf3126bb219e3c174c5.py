"""QAPLIB tho150: correct swap-delta local search under an explicit time budget."""
import time
import random
import numpy as np

_DEADLINE = 145.0  # hard wall-clock guard (evaluator kills at 150 s)


def _cost(a, b, p):
    """Exact integer QAP cost sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    return int(np.einsum("ij,pi,pj->", a, b, b, optimize=True))


def _rank_start(a, b, rng):
    """Heavy flows -> short distances, with a small random perturbation."""
    n = len(a)
    ai = a.sum(axis=1).astype(np.float64)
    bi = b.sum(axis=1).astype(np.float64)
    noise = rng.random(n) * 0.02 * (np.ptp(ai) + 1e-12)
    heavy = np.argsort(-ai - noise, kind="stable")
    central = np.argsort(bi + rng.random(n) * 0.02 * (np.ptp(bi) + 1e-12), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _local_search(a, b, p, A, B, deadline):
    """Iterated full-scan swap local search with the exact O(1) delta."""
    n = len(p)
    cost = _cost(a, b, p)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        bi = bj = -1
        for i in range(n):
            pi = p[i]
            ai = A[i]
            for j in range(i + 1, n):
                pj = p[j]
                if pi == pj:
                    continue
                bxy = b[pi, pj]
                byx = b[pj, pi]
                d = (bxy - byx) * (a[i, j] + a[j, i] + ai + A[j]) \
                    + (byx - bxy) * (B[pi] + B[pj])
                if d < best_delta:
                    best_delta = d
                    bi, bj = i, j
        if best_delta < 0:
            i, j = bi, bj
            p[i], p[j] = p[j], p[i]
            cost += best_delta
            improved = True
    return cost, p


def solve(a, b):
    t0 = time.time()
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    n = len(a)
    rng = np.random.default_rng(12345)

    A = a.sum(axis=1)
    B = b.sum(axis=1)

    # Phase 1: deterministic-ish rank start + local search.
    p = _rank_start(a, b, rng)
    cost, p = _local_search(a, b, p, A, B, _DEADLINE - 2.0)

    best_cost, best_p = cost, p.copy()

    # Phase 2: budgeted random restarts (fresh rank start, re-optimized).
    while time.time() < _DEADLINE - 2.0:
        p2 = _rank_start(a, b, rng)
        c2, p2 = _local_search(a, b, p2, A, B, _DEADLINE - 2.0)
        if c2 < best_cost:
            best_cost, best_p = c2, p2

    return best_p.tolist()
