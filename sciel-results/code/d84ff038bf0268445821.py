"""QAPLIB tho150 solver: fast delta-swap simulated annealing + greedy init + polish.
SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 116.0  # safety margin under the 120 s hard limit


def _qap_cost(a, b, p):
    """Exact integer QAP objective for permutation p (facility i at location p[i])."""
    ap = a[np.ix_(p, p)]
    return int(np.sum(ap * b))


def _init_solution(a, b, n, rng):
    """Greedy insertion start: rank facilities by flow weight, insert each into the
    location minimizing the incremental cost. O(n^2) total, cheap for n=150."""
    a_s = (a + a.T) * 0.5
    b_s = (b + b.T) * 0.5
    w = a_s.sum(axis=1)
    order = np.argsort(-w, kind="stable")
    placed = np.zeros(n, dtype=np.int64)
    p = np.zeros(n, dtype=np.int64)
    locs = np.arange(n, dtype=np.int64)
    for k in range(n):
        i = order[k]
        if k == 0:
            p[i] = 0
            locs = locs[1:]
            continue
        # incremental cost of placing i at each free location j
        inc = b_s[i, locs] + b_s[locs, i]
        # interaction with already-placed facilities
        idx = np.where(placed)[0]
        for ii in idx:
            inc = inc + a_s[i, ii] * b_s[p[ii], locs]
        best = locs[int(np.argmin(inc))]
        p[i] = best
        placed[i] = 1
        locs = np.delete(locs, np.where(locs == best)[0][0])
    return p


def _delta_swap(a, b, res, i, j, pi, pj):
    """O(n) cost delta for swapping the locations of facilities i and j.
    res[j] = sum_k a[i, k] * b[p[k], j].  No array copies, no redundant dots."""
    a_ij = a[i, j]
    b_ip = b[pi, pj]
    b_jp = b[pj, pi]
    r_i = res[i]
    r_j = res[j]
    # new contribution of i,j pair (symmetric) and their coupling to others
    return (a_ij * (b_ip + b_jp - 2 * r_i - 2 * r_j)
            + (r_i - r_j) * (b_jp - b_ip))


def _sa(a, b, p, n, rng, t_start, deadline, T0, T1, iters):
    """Simulated annealing with O(n) delta swaps. Returns (p, best_cost)."""
    res = a @ b[:, p]  # res[j] = sum_i a[i, p[i]] * b[p[i], j]
    cost = int(res.sum())
    best_cost = cost
    best_p = p.copy()
    T = T0
    cool = (T1 / T0) ** (1.0 / max(1, iters))
    for _ in range(iters):
        if time.time() - t_start > deadline:
            break
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        pi = int(p[i])
        pj = int(p[j])
        d = _delta_swap(a, b, res, i, j, pi, pj)
        if d < 0 or rng.random() < np.exp(-d / T):
            # apply swap
            p[i], p[j] = pj, pi
            cost += d
            # update res in place
            ai = a[i]
            bi = b[pi]
            bj = b[pj]
            res[pi] = res[pi] + d - a[i, i] * b[pi, pi]
            res[pj] = res[pj] + d - a[j, j] * b[pj, pj]
            # recompute affected rows exactly (only i and j change)
            res[i] = ai @ b[:, p]
            res[j] = a[j] @ b[:, p]
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        T *= cool
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + _DEADLINE
    rng = random.Random(1234567)
    best_p = None
    best_cost = None

    # Multi-restart SA with adaptive cooling, each restart seeded from the best so far.
    iters = 60000
    T0 = 5000.0
    T1 = 1.0
    restarts = 4
    for r in range(restarts):
        if time.time() - t_start > deadline:
            break
        if r == 0:
            p = _init_solution(a, b, n, rng)
        else:
            p = best_p.copy()
        # progressively smaller T0 for later (polishing) restarts
        p, c = _sa(a, b, p, n, rng, t_start, deadline, T0 / (1 + 0.5 * r), T1, iters)
        if best_cost is None or c < best_cost:
            best_cost = c
            best_p = p

    # Final high-precision 2-opt polish: full sweep, accept any improving swap.
    if time.time() - t_start < deadline:
        res = a @ b[:, best_p]
        cost = int(res.sum())
        improved = True
        while improved and time.time() - t_start < deadline:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    pi = int(best_p[i])
                    pj = int(best_p[j])
                    d = _delta_swap(a, b, res, i, j, pi, pj)
                    if d < 0:
                        best_p[i], best_p[j] = pj, pi
                        cost += d
                        res[pi] = res[pi] + d - a[i, i] * b[pi, pi]
                        res[pj] = res[pj] + d - a[j, j] * b[pj, pj]
                        res[i] = a[i] @ b[:, best_p]
                        res[j] = a[j] @ b[:, best_p]
                        improved = True
        best_cost = cost

    if best_p is None:
        best_p = _init_solution(a, b, n, rng)
        best_cost = _qap_cost(a, b, best_p)
    return best_p.tolist()
# EVOLVE-BLOCK-END
