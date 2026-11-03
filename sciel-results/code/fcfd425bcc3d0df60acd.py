"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under the 120 s limit

    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    bT = b.T.copy()

    # ---- initial solution: rank-matching (heavy flows -> short distances) ----
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    inv = np.empty(n, dtype=np.int64)
    inv[p] = np.arange(n)

    # ---- fast cost computation ----
    def full_cost():
        c = 0
        for i in range(n):
            c += int(np.dot(a[i], b[p[i]]))
        return c

    cur = full_cost()
    best = cur
    best_p = p.copy()

    # ---- precompute for fast delta ----
    # For a swap of facilities i and j (at locations u=p[i], v=p[j]):
    # delta = sum_k (a[i,k] - a[j,k]) * (b[k,v] - b[k,u]) + (a[i,v]+a[j,v]-a[i,u]-a[j,u])*b[u,v]
    # We can precompute row sums to help with candidate selection.

    a_rowsum = a.sum(axis=1)
    b_colsum = b.sum(axis=1)  # sum over rows for each column (location)

    # ---- 2-opt local search with O(n) delta per swap ----
    # We iterate through all pairs (i, j) and pick the best improving swap.
    # To speed up, we focus on "high-impact" facilities first.

    # Sort facilities by their row-sum of a (descending) - these have biggest impact
    order = np.argsort(-a_rowsum, kind="stable")

    it = 0
    no_improve = 0
    max_no_improve = 5  # restart after this many passes without improvement

    while time.time() < deadline:
        improved = False

        # Iterate through pairs in order of impact
        for ii in range(n):
            if time.time() > deadline - 2.0:
                break
            i = order[ii]
            u = p[i]
            ai = a[i]

            for jj in range(ii + 1, n):
                j = order[jj]
                v = p[j]
                if u == v:
                    continue
                aj = a[j]

                # Compute delta for swapping i and j
                # delta = sum_k (a[i,k] - a[j,k]) * (b[k,v] - b[k,u]) + (a[i,v]+a[j,v]-a[i,u]-a[j,u])*b[u,v]
                diff_b = bT[v] - bT[u]  # shape (n,)
                diff_a = ai - aj        # shape (n,)
                cross = int(np.dot(diff_a, diff_b))
                delta = int((ai[v] + aj[v] - ai[u] - aj[u]) * b[u, v]) + cross

                if delta < 0:
                    # Accept the swap
                    p[i], p[j] = v, u
                    inv[u], inv[v] = j, i
                    cur += delta
                    u, v = v, u  # swap back for consistency
                    improved = True

                    if cur < best:
                        best = cur
                        best_p = p.copy()

        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                # Perturb: random restart with small random changes
                if time.time() > deadline - 5.0:
                    break

                # Apply a few random swaps to escape local optimum
                for _ in range(min(10, n // 10)):
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i == j:
                        continue
                    u = p[i]
                    v = p[j]
                    if u == v:
                        continue
                    aj = a[j]
                    ai = a[i]
                    diff_b = bT[v] - bT[u]
                    diff_a = ai - aj
                    cross = int(np.dot(diff_a, diff_b))
                    delta = int((ai[v] + aj[v] - ai[u] - aj[u]) * b[u, v]) + cross

                    p[i], p[j] = v, u
                    inv[u], inv[v] = j, i
                    cur += delta

                    if cur < best:
                        best = cur
                        best_p = p.copy()

                no_improve = 0
        else:
            no_improve = 0

        it += 1

    return best_p.tolist()
# EVOLVE-BLOCK-END
