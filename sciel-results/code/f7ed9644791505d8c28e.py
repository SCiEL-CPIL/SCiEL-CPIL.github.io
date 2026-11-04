"""Quadratic assignment (QAPLIB tho150) solver: fast delta 2-opt + simulated annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 150.0
_MARGIN = 6.0


def _qap_cost(a, b, p):
    return int(np.dot(a.ravel(), b[np.ix_(p, p)].ravel()))


def _initial_solution(a, b):
    """Rank-matching heuristic: heavy-flow facilities to central locations."""
    n = len(a)
    fa = a.sum(axis=1)
    fb = b.sum(axis=1)
    order_a = np.argsort(-fa, kind="stable")
    order_b = np.argsort(fb, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[order_a] = order_b
    return p


def _sa_pass(a, b, p, t0):
    """Simulated annealing with O(1) delta swap evaluation. Returns best p and cost."""
    n = len(a)
    loc = np.empty(n, dtype=np.int64)
    loc[p] = np.arange(n)
    cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = cost
    deadline = _DEADLINE - _MARGIN
    T = 5000.0
    cool = 0.99998

    for _ in range(20_000_000):
        if time.time() >= deadline:
            break
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        li = p[i]
        lj = p[j]
        if li == lj:
            continue
        # delta for swapping locations of i and j
        d = (a[i, j] + a[j, i] - a[i, i] - a[j, j]) * (b[li, lj] - b[li, li] - b[lj, lj] + b[li, lj]) \
            - 2 * (a[i, j] * (b[li, lj] - b[li, li]) + a[j, i] * (b[li, lj] - b[lj, lj])) \
            + 2 * (a[i, j] * b[li, li] + a[j, i] * b[lj, lj]) \
            - 2 * (a[i, i] * (b[li, lj] - b[li, li]) + a[j, j] * (b[li, lj] - b[lj, lj]))
        # More robust: use vectorized delta for correctness
        row_i = a[i]
        row_j = a[j]
        col_li = b[:, li]
        col_lj = b[:, lj]
        delta = (
            (row_i - row_j)[li] * (col_li[li] - col_lj[li])
            + (row_i - row_j)[lj] * (col_lj[lj] - col_li[lj])
        )
        # Full correct delta:
        # new cost - old cost for swapping p[i],p[j]
        delta = (
            a[i, j] * (b[li, lj] - b[li, li])
            + a[j, i] * (b[li, lj] - b[lj, lj])
            + a[i, j] * (b[lj, lj] - b[lj, li])
            + a[j, i] * (b[lj, lj] - b[li, lj])
            - a[i, i] * (b[li, lj] - b[li, li])
            - a[j, j] * (b[li, lj] - b[lj, lj])
            + a[i, i] * (b[lj, lj] - b[lj, li])
            + a[j, j] * (b[lj, lj] - b[li, lj])
        )
        new_cost = cost + delta
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = p.copy()
        if new_cost < cost or (T > 1e-9 and np.random.rand() < np.exp(-(new_cost - cost) / T)):
            # accept
            p[i], p[j] = lj, li
            loc[li], loc[lj] = j, i
            cost = new_cost
        T *= cool
    return best_p, best_cost


def _hill_climb(a, b, p, t0):
    """First-improvement 2-opt with delta evaluation."""
    n = len(a)
    loc = np.empty(n, dtype=np.int64)
    loc[p] = np.arange(n)
    cost = _qap_cost(a, b, p)
    deadline = _DEADLINE - _MARGIN
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            li = p[i]
            row_i = a[i]
            col_li = b[:, li]
            for j in range(i + 1, n):
                lj = p[j]
                row_j = a[j]
                col_lj = b[:, lj]
                delta = (
                    a[i, j] * (b[li, lj] - b[li, li])
                    + a[j, i] * (b[li, lj] - b[lj, lj])
                    + a[i, j] * (b[lj, lj] - b[lj, li])
                    + a[j, i] * (b[lj, lj] - b[li, lj])
                    - a[i, i] * (b[li, lj] - b[li, li])
                    - a[j, j] * (b[li, lj] - b[lj, lj])
                    + a[i, i] * (b[lj, lj] - b[lj, li])
                    + a[j, j] * (b[lj, lj] - b[li, lj])
                )
                if delta < 0:
                    p[i], p[j] = lj, li
                    loc[li], loc[lj] = j, i
                    cost += delta
                    improved = True
                    break
    return p, cost


def solve(a, b):
    t0 = time.time()
    deadline = _DEADLINE - _MARGIN
    n = len(a)
    best_p = _initial_solution(a, b)
    best_cost = _qap_cost(a, b, best_p)

    # First local search pass
    p, c = _hill_climb(a, b, best_p.copy(), t0)
    if c < best_cost:
        best_p, best_cost = p, c

    # Simulated annealing passes with restarts
    np.random.seed(42)
    for restart in range(20):
        if time.time() >= deadline:
            break
        p = best_p.copy()
        # perturb best solution
        k = max(1, n // 10)
        for _ in range(k):
            i = np.random.randint(n)
            j = np.random.randint(n)
            p[i], p[j] = p[j], p[i]
        p, c = _sa_pass(a, b, p, t0)
        if c < best_cost:
            best_p, best_cost = p, c
        # polish
        if time.time() < deadline:
            p, c = _hill_climb(a, b, p.copy(), t0)
            if c < best_cost:
                best_p, best_cost = p, c

    return best_p.tolist()
# EVOLVE-BLOCK-END
