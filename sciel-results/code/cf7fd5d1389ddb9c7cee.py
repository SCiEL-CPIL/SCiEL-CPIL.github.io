"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110.0  # safety margin under the 120 s limit

    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    bT = b.T

    # ---- initial solution: rank-matching (heavy flows -> short distances) ----
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    inv = np.empty(n, dtype=np.int64)
    inv[p] = np.arange(n)

    # ---- fast cost + per-row aggregate ----
    # cost = sum_{i,j} a[i,j]*b[p[i],p[j]]
    # row[i] = sum_j a[i,j]*b[p[i],p[j]]  (the "row contribution" of facility i)
    def full_cost():
        c = 0
        for i in range(n):
            c += int(np.dot(a[i], b[p[i]]))
        return c

    cur = full_cost()

    # ---- tabu search ----
    tabu_len = 18
    tabu = {}  # facility i -> iteration until which re-placing i to a loc is tabu
    best = cur
    best_p = p.copy()
    it = 0

    # precompute per-facility row-sum of a to rank candidate swaps
    a_rowsum = a.sum(axis=1)

    while time.time() < deadline:
        # build candidate list: top facilities by a_rowsum (most impactful)
        cand = np.argsort(-a_rowsum, kind="stable")
        best_delta = 0
        best_i = -1
        best_j = -1
        best_asp = False

        for i in cand:
            if tabu.get(i, -1) >= it:
                continue
            u = p[i]
            ai = a[i]
            for j in range(n):
                if j == i:
                    continue
                v = p[j]
                if v == u:
                    continue
                # aspiration: allow if it beats best
                # compute delta in O(n)
                aj = a[j]
                # sum_k (a_i,k - a_j,k)*(b[k,v]-b[k,u])
                diff_b = bT[v] - bT[u]      # vector over k
                diff_a = ai - aj            # vector over k
                cross = int(np.dot(diff_a, diff_b))
                delta = int((ai[v] + aj[v] - ai[u] - aj[u]) * b[u, v]) + cross
                tabu_j = tabu.get(j, -1) >= it
                if delta < best_delta or (delta < best and not tabu_j):
                    # aspiration: accept tabu move if it improves global best
                    if delta < best:
                        best_delta = delta
                        best_i, best_j = i, j
                        best_asp = True
                    elif not tabu_j:
                        best_delta = delta
                        best_i, best_j = i, j
                        best_asp = False

        if best_i < 0:
            break

        i, j = best_i, best_j
        u, v = p[i], p[j]
        p[i], p[j] = v, u
        inv[u], inv[v] = j, i

        cur += best_delta
        if cur < best:
            best = cur
            best_p = p.copy()

        tabu[i] = it + tabu_len
        tabu[j] = it + tabu_len
        it += 1

    return best_p.tolist()
# EVOLVE-BLOCK-END
