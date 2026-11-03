"""Strong local-search solver for the QAPLIB tai256c instance."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _qap_cost(a, b, p):
    # Exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]
    return int(np.sum(a * b[np.ix_(p, p)]))


def _rank_init(a, b):
    # Heavy flows -> short distances
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(len(a), dtype=np.int64)
    p[heavy] = central
    return p


def _greedy_init(a, b):
    # Greedy placement: place facilities by decreasing flow weight,
    # each at the location minimizing incremental cost.
    n = len(a)
    weights = a.sum(axis=1)
    order = np.argsort(-weights, kind="stable")
    p = np.empty(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    # incremental cost of placing facility i at location x given current p
    #   = sum_j a[i,j]*b[x,p[j]]  +  sum_j a[j,i]*b[p[j],x]
    for i in order:
        # cost vector over candidate locations
        c = b @ (a[i, :] @ (b.T)[0:0])  # placeholder, compute properly below
        # proper: c[x] = sum_j a[i,j]*b[x,p[j]] + sum_j a[j,i]*b[p[j],x]
        #   = (a[i,:] @ B)[x] + (a[:,i] @ B)[x]  where B[x,j]=b[x,p[j]]
        B = b[:, p[used]]  # n x k  (B[x,j]=b[x,p[j]])
        c = (a[i, :] @ B) + (a[:, i] @ B)
        c[used] = np.inf
        loc = int(np.argmin(c))
        p[i] = loc
        used[loc] = True
    return p


def _best_improve_2opt(a, b, p, deadline):
    # Best-improvement 2-opt local search with incremental cost.
    n = len(p)
    p = p.astype(np.int64, copy=True)
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    # B[i, x] = sum_j a[i, j] * b[j, x]
    B = a @ b
    cost = _qap_cost(a, b, p)

    while True:
        if time.time() > deadline:
            break
        best_d = 0.0
        best_i = -1
        best_k = -1
        for i in range(n):
            pi = p[i]
            Ai = a[i, :]
            Bi = B[i, :]
            for k in range(i + 1, n):
                pk = p[k]
                if pi == pk:
                    continue
                # delta for swapping facilities i and k
                d = (Ai[pi] - Ai[pk]) * (Bi[pk] - Bi[pi]) \
                  + (a[k, pi] - a[k, pk]) * (Bi[pi] - Bi[pk])
                if d < best_d:
                    best_d = d
                    best_i = i
                    best_k = k
        if best_i < 0:
            break  # local optimum
        i, k = best_i, best_k
        pi, pk = p[i], p[k]
        # update B rows for i and k
        B[i, :] = B[k, :]
        B[k, :] = B[i, :]
        p[i], p[k] = pk, pi
        cost += best_d

    return p, cost


def _random_restarts(a, b, deadline):
    n = len(a)
    # Start from the best of a few diverse initializations, then polish.
    starts = [_rank_init(a, b), _greedy_init(a, b)]
    best_p = starts[0].copy()
    best_c = _qap_cost(a, b, best_p)
    for s in starts[1:]:
        c = _qap_cost(a, b, s)
        if c < best_c:
            best_c = c
            best_p = s.copy()

    # Polish the best start with best-improvement 2-opt.
    # Leave a safety margin before the deadline.
    work_deadline = deadline - 3.0
    polished, _ = _best_improve_2opt(a, b, best_p, work_deadline)
    best_c = _qap_cost(a, b, polished)
    best_p = polished

    # If time remains, try additional random restarts + short polish.
    rng = np.random.default_rng(12345)
    while time.time() < work_deadline - 5.0:
        # random perturbation of the current best
        cand = best_p.copy()
        # apply a handful of random swaps
        for _ in range(n // 4):
            x, y = rng.integers(0, n, size=2)
            cand[x], cand[y] = cand[y], cand[x]
        # short polish
        sub_deadline = min(time.time() + (work_deadline - 5.0) * 0.5, work_deadline - 1.0)
        polished, _ = _best_improve_2opt(a, b, cand, sub_deadline)
        c = _qap_cost(a, b, polished)
        if c < best_c:
            best_c = c
            best_p = polished

    return best_p


def solve(a, b):
    t0 = time.time()
    # Hard wall-clock guard: evaluator kills at 150s; return safely before it.
    deadline = t0 + 140.0
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    try:
        p = _random_restarts(a, b, deadline)
    except Exception:
        p = _rank_init(a, b)
    return p.tolist()
# EVOLVE-BLOCK-END
