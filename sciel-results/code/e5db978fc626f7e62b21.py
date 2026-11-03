"""QAPLIB tho150: fast incremental 2-opt local search (correct swap delta).

Cost = sum_{i,j} a[i,j] * b[p[i], p[j]].  A swap of the locations assigned to
facilities i and j changes only the terms touching rows/cols i and j, so the
cost delta is an O(n) computation.  This lets us run a very large number of
2-opt moves within the wall-clock budget.  Multiple restarts (rank-matching +
random) help escape shallow local optima.
"""
import time

import numpy as np


def _initial_solution(a, b):
    n = a.shape[0]
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _random_solution(a, rng):
    n = a.shape[0]
    return rng.permutation(n).astype(np.int64)


def _swap_delta(a, b, p, i, j):
    """Cost delta of swapping the locations assigned to facilities i and j."""
    pi = p[i]
    pj = p[j]
    if pi == pj:
        return 0
    # Terms that change: those with a row or column in {i, j}.
    # Current contribution involving i or j:
    #   cur = a[i,:].b[b[pi, p]] + a[:,i].b[b[p, pi]] + a[j,:].b[b[pj,p]] + a[:,j].b[b[p,pj]]
    # New (swap pi<->pj for i and j):
    #   new = a[i,:].b[b[pj, p']] + a[:,i].b[b[p', pj]] + a[j,:].b[b[pi, p']] + a[:,j].b[b[p', pi]]
    # where p' is p with i->pj, j->pi.  The difference is O(n).
    # Use vectorized inner products.
    ai = a[i]
    aj = a[j]
    aiT = a[:, i]
    ajT = a[:, j]

    # For each facility k, the location p[k] is unchanged except k=i (->pj) and k=j (->pi).
    # We compute the sum over k of a[i,k]*b[pi,p[k]] etc.
    # Build the "current" and "new" b-rows indexed by p.
    # cur_i = sum_k a[i,k] b[pi, p[k]] = ai @ b[pi, p]
    # new_i = sum_k a[i,k] b[pj, p[k]] but with p[i]=pj: for k=i, b[pj, pj]; others b[pj, p[k]]
    #        = ai @ b[pj, p] - ai[i]*b[pj, p[i]] + ai[i]*b[pj, pj]
    # Simpler: compute full sums then correct the two diagonal terms.

    b_pi = b[pi]
    b_pj = b[pj]

    cur_i = float(ai @ b_pi[p])
    cur_j = float(aj @ b_pj[p])
    cur_iT = float(aiT @ b_pi[p])  # a[:,i] . b[pi, p] = sum_k a[k,i] b[pi, p[k]]
    cur_jT = float(ajT @ b_pj[p])

    # After swap, i uses pj, j uses pi.
    # new_i = sum_k a[i,k] b[pj, p'[k]] where p'[i]=pj, p'[j]=pi, else p[k]
    #       = sum_k a[i,k] b[pj, p[k]] - a[i,i] b[pj, p[i]] - a[i,j] b[pj, p[j]]
    #             + a[i,i] b[pj, pj] + a[i,j] b[pj, pi]
    new_i = float(ai @ b_pj[p]) - ai[i] * b[pj, pi] - ai[j] * b[pj, pj] + ai[i] * b[pj, pj] + ai[j] * b[pj, pi]
    # simplify: -ai[i]*b[pj,pi] + ai[i]*b[pj,pj] + (-ai[j]*b[pj,pj] + ai[j]*b[pj,pi])
    new_i = float(ai @ b_pj[p]) + ai[i] * (b[pj, pj] - b[pj, pi]) + ai[j] * (b[pj, pi] - b[pj, pj])
    # = ai @ b_pj[p] + (ai[i]-ai[j])*(b[pj,pj]-b[pj,pi])

    new_j = float(aj @ b_pi[p]) + (aj[j] - aj[i]) * (b[pi, pi] - b[pi, pj])

    # Transpose terms: cur_iT = sum_k a[k,i] b[pi, p[k]]; new_iT uses p' and b[pj, p'[k]]
    new_iT = float(aiT @ b_pj[p]) + (a[i, i] - a[j, i]) * (b[pj, pj] - b[pj, pi])
    new_jT = float(ajT @ b_pi[p]) + (a[i, j] - a[j, j]) * (b[pi, pi] - b[pi, pj])

    delta = (new_i + new_j + new_iT + new_jT) - (cur_i + cur_j + cur_iT + cur_jT)
    return delta


def _local_search(a, b, p, deadline_guard, rng):
    n = a.shape[0]
    cost = int(np.einsum("ij,ik->", a, b[np.arange(n), p]))
    # cost = sum_{i,j} a[i,j] b[p[i], p[j]] ; einsum "ij,ik->" with b[p[i],p[j]]
    # Build the permuted b matrix once for the initial cost.
    cost = int(np.einsum("ij,ij->", a, b[np.arange(n)][:, p]))

    best_p = p.copy()
    best_cost = cost

    while True:
        if time.perf_counter() > deadline_guard:
            break
        improved = False
        order = rng.permutation(n)
        for ii in range(n):
            if time.perf_counter() > deadline_guard:
                break
            i = order[ii]
            for jj in range(ii + 1, n):
                j = order[jj]
                if p[i] == p[j]:
                    continue
                d = _swap_delta(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += int(d)
                    improved = True
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
        if not improved:
            break
    return best_p.tolist(), best_cost


def solve(a, b):
    start = time.perf_counter()
    deadline = start + 140.0
    deadline_guard = deadline - 1.0
    rng = np.random.default_rng(12345)

    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)

    best_p, best_cost = _local_search(a, b, _initial_solution(a, b), deadline_guard, rng)

    while time.perf_counter() < deadline - 2.0:
        p0 = _random_solution(a, rng)
        sub = min(deadline_guard, time.perf_counter() + 25.0)
        p, c = _local_search(a, b, p0, sub, rng)
        if c < best_cost:
            best_cost = c
            best_p = p

    return best_p
