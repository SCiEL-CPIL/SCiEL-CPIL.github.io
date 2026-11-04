"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _compute_cost(a, b, p):
    # exact integer cost sum_{i,j} a[i,j]*b[p[i],p[j]]
    return int(np.sum(a * b[np.ix_(p, p)]))


def solve(a, b):
    n = a.shape[0]
    deadline = time.time() + 150.0 - 3.0  # safety margin before the 150s kill

    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)

    a_row = a.sum(axis=1)
    b_row = b.sum(axis=1)

    # ---- initial solution: rank matching (heavy flows -> short distances) ----
    heavy = np.argsort(-a_row, kind="stable")
    central = np.argsort(b_row, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    cur_cost = _compute_cost(a, b, p)

    best_p = p.copy()
    best_cost = cur_cost

    # ---- precompute the "hot" candidate set (largest a row-sums) ----
    # swapping two heavy facilities dominates most single-swap improvements
    k = min(n, 60)
    hot = np.argsort(-a_row, kind="stable")[:k]

    # ---- simulated annealing with vectorized neighborhood ----
    T = 10000.0
    T_min = 1.0
    cooling = 0.995

    rng = np.random.default_rng(12345)

    while time.time() < deadline:
        if T <= T_min:
            # occasional gentle restart to escape deep local minima
            if cur_cost > best_cost * 1.05:
                p = best_p.copy()
                cur_cost = best_cost
            T = max(T_min, T * 0.5)

        # draw a random pair of hot facilities to swap
        i = hot[rng.integers(0, k)]
        j = hot[rng.integers(0, k)]
        if i == j:
            continue

        pi = p[i]
        pj = p[j]
        if pi == pj:
            continue

        # exact O(1) cost delta for swapping facilities i and j
        # delta = (a[i,j]+a[j,i])*(b[pi,pj]-b[pj,pi]) + (a[i,i]-a[j,j])*(b[pi,pi]-b[pj,pj])
        delta = (a[i, j] + a[j, i]) * (b[pi, pj] - b[pj, pi]) \
            + (a[i, i] - a[j, j]) * (b[pi, pi] - b[pj, pj])

        if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
            p[i], p[j] = pj, pi
            cur_cost += delta
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_p = p.copy()

        T *= cooling

    return best_p.tolist()
# EVOLVE-BLOCK-END
