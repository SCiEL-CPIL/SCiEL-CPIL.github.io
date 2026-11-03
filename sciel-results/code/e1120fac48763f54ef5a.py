"""Local search + LAP-guided perturbation for QAP wil100."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def solve(a, b):
    n = a.shape[0]
    start = time.time()
    deadline = start + 138.0  # safety margin under 150s wall-clock

    # ---- rank-matching initial solution ----
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    best_p = p.copy()
    best_c = _cost(a, b, p)

    # ---- precompute flow-weighted distance matrix C and incoming-flow g ----
    # C[i, l] = sum_j a[i, j] * b[l, p[j]]  (cost of putting facility i at location l)
    # g[l]    = sum_j a[p[j], i] ... maintained dynamically
    C = np.dot(a, b[:, p])          # C[i, l] = sum_j a[i,j]*b[l, p[j]]
    # g[l] = sum_j a[p[j], l] where the "l" here is the *location index* in b,
    # i.e. g[l] = sum_j a[p[j], l].  (column l of a indexed by current placement)
    g = a[:, p].sum(axis=0)         # g[l] = sum_j a[p[j], l]

    def swap_delta(i, j):
        # delta in cost of swapping facilities i and j (locations p[i], p[j])
        li = p[i]
        lj = p[j]
        return (
            C[i, lj] + C[j, li] - C[i, li] - C[j, lj]
            + g[li] + g[lj] - 2 * a[i, j]
        )

    def do_swap(i, j):
        nonlocal best_c
        li = p[i]
        lj = p[j]
        # update C rows for i and j
        C[i, :] = C[j, :]
        C[j, :] = C[i, :]
        # update g at li and lj
        g[li] += a[i, j] - a[j, i]
        g[lj] += a[j, i] - a[i, j]
        p[i] = lj
        p[j] = li

    def local_search(p, c):
        nonlocal best_c
        improved = True
        while improved:
            if time.time() > deadline:
                return p, c
            improved = False
            # random order to break ties
            order = np.random.permutation(n)
            for i in order:
                if time.time() > deadline:
                    return p, c
                li = p[i]
                # find best j != i
                best_j = -1
                best_d = 0
                for j in order:
                    if j == i:
                        continue
                    lj = p[j]
                    if lj == li:
                        continue
                    d = swap_delta(i, j)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j >= 0 and best_d < 0:
                    do_swap(i, best_j)
                    c += best_d
                    improved = True
                    if c < best_c:
                        best_c = c
                        best_p = p.copy()
        return p, c

    p, best_c = local_search(p, best_c)

    # ---- LAP-guided perturbation loop ----
    # Build a smooth surrogate: minimize sum_i C[i, p[i]] + sum_j g-like term.
    # We approximate the QAP by the linear surrogate sum_i (sum_j a[i,j]) * dist(p[i], p[j])
    # which is captured by C.  Solve LAP on C to get a globally consistent reassignment.
    rng = np.random.default_rng(0)
    while time.time() < deadline - 5.0:
        # solve LAP on C (linear surrogate)
        C_sym = (C + C.T) / 2.0  # symmetric surrogate
        # C is n x n: rows = facilities, cols = locations.  LAP assigns facility->location.
        rows, cols = linear_sum_assignment(C_sym)
        new_p = np.empty(n, dtype=np.int64)
        new_p[rows] = cols
        # rebuild C and g for new_p
        C = np.dot(a, b[:, new_p])
        g = a[:, new_p].sum(axis=0)
        c = _cost(a, b, new_p)
        # polish with local search
        new_p, c = local_search(new_p, c)
        if c < best_c:
            best_c = c
            best_p = new_p.copy()
            C = np.dot(a, b[:, best_p])
            g = a[:, best_p].sum(axis=0)
        # occasionally perturb with random swaps to escape
        if time.time() < deadline - 5.0:
            for _ in range(rng.integers(2, 6)):
                i = rng.integers(n)
                j = rng.integers(n)
                if i != j:
                    li = best_p[i]
                    lj = best_p[j]
                    d = (C[i, lj] + C[j, li] - C[i, li] - C[j, lj]
                         + g[li] + g[lj] - 2 * a[i, j])
                    if d < 0:
                        # apply swap to best_p and update C, g
                        C[i, :] = C[j, :]
                        C[j, :] = C[i, :]
                        g[li] += a[i, j] - a[j, i]
                        g[lj] += a[j, i] - a[i, j]
                        best_p[i] = lj
                        best_p[j] = li
                        best_c += d

    # ---- final high-precision polish ----
    if time.time() < deadline - 2.0:
        C = np.dot(a, b[:, best_p])
        g = a[:, best_p].sum(axis=0)
        best_p, best_c = local_search(best_p, best_c)

    return best_p.tolist()
# EVOLVE-BLOCK-END
