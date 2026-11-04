"""Sinkhorn-annealed QAP with Hungarian extraction and swap polish."""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

_DEADLINE_FRAC = 0.92


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _sinkhorn_extract(a, b, n, iters, eps, t0):
    """Sinkhorn-Knopp on a doubly-stochastic matrix; returns (perm, cost, X)."""
    # objective on X: sum_ij a_ij * (b^T)_{ij} * X_ij  with C = a * b^T
    C = a * b.T
    # temperature/regularization annealing: start hot (explore), cool (lock in)
    lam0 = max(float(np.abs(C).max()), 1.0)
    X = np.full((n, n), 1.0 / n, dtype=np.float64)
    last_cost = None
    last_X = X.copy()
    last_p = None
    for it in range(iters):
        if time.time() > t0:
            break
        lam = lam0 * (0.2 + 0.8 * (it / max(1, iters - 1)))  # cool toward low temp
        # log-sum-exp row/col normalization of exp(-C/lam)
        W = np.exp(-C / lam)
        for _ in range(2):
            r = W.sum(axis=1, keepdims=True)
            r[r == 0] = 1.0
            W = W / r
            c = W.sum(axis=0, keepdims=True)
            c[c == 0] = 1.0
            W = W / c
        # normalize to doubly stochastic (sum 1)
        s = W.sum()
        if s > 0:
            W = W / s
        # extract a permutation from the current matrix
        p = linear_sum_assignment(-W)
        p = p[1]
        cost = _qap_cost(a, b, p)
        last_cost = cost
        last_X = W.copy()
        last_p = p
    return last_p, last_cost, last_X


def _swap_polish(a, b, p, n, t0):
    """Exact 2-opt swap local search with deadline guard."""
    p = p.copy()
    cur = _qap_cost(a, b, p)
    # precompute per-facility row of a and b for fast delta
    # delta of swapping p[i],p[j]:
    #   new = cur - 2*sum_k a[i,k]*b[p[i],p[k]] ... use incremental
    # We use a vectorized delta for a pair.
    improved = True
    while improved and time.time() < t0:
        improved = False
        # iterate over pairs in a deterministic order; compute delta for all j for a given i
        for i in range(n):
            if time.time() > t0:
                break
            pi = p[i]
            # contribution involving facility i and location pi
            # For each j, swapping p[i]<->p[j] changes cost by:
            #   d = 2*( a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] - a[i,j]*b[pi,pj] ) ... derive below
            # Use direct recomputation of affected terms via precomputed row sums.
            # Precompute S_i[j] = sum_k a[i,k]*b[pi,p[k]]  (cost of i at pi vs all)
            # We'll compute delta for pair (i,j) exactly:
            #   affected locations: pi, pj ; affected facilities: i, j
            #   old terms: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] + a[i,i]*b[pi,pi] + a[j,j]*b[pj,pj]
            #   new terms: a[i,j]*b[pj,pj] + a[j,i]*b[pi,pi] + a[i,i]*b[pj,pi] + a[j,j]*b[pi,pj]
            # plus cross with all other k (k not i,j):
            #   old: a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pi] + a[j,k]*b[pj,p[k]] + a[k,j]*b[p[k],pj]
            #   new: a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj] + a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi]
            # Vectorize over k.
            for j in range(i + 1, n):
                if time.time() > t0:
                    break
                pj = p[j]
                if pi == pj:
                    continue
                # cross terms over all k
                old_cross = np.sum(a[i, :] * b[pi, p]) + np.sum(a[:, i] * b[p, pi]) \
                          + np.sum(a[j, :] * b[pj, p]) + np.sum(a[:, j] * b[p, pj])
                new_cross = np.sum(a[i, :] * b[pj, p]) + np.sum(a[:, i] * b[p, pj]) \
                          + np.sum(a[j, :] * b[pi, p]) + np.sum(a[:, j] * b[p, pi])
                # self/diagonal terms (i,j) are included in the above since k ranges all;
                # but we must be careful: the above includes k=i and k=j correctly because
                # p is a permutation so b[pi,p] and b[p,pi] index all locations.
                # However the (i,i) and (j,j) facility self-terms are already in a[i,:]*b[pi,p].
                # So total delta:
                d = (new_cross - old_cross)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    cur += d
                    improved = True
                    if time.time() > t0:
                        return p, cur
    return p, cur


def solve(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    n = a.shape[0]
    start = time.time()
    t0 = start + 120.0 * _DEADLINE_FRAC

    # 1) Sinkhorn-annealed continuous relaxation -> initial permutation
    iters = 60
    p, cost, X = _sinkhorn_extract(a, b, n, iters, 1.0, t0)

    best_p = p.copy()
    best_cost = cost

    # 2) Polish with exact swap local search
    p2, c2 = _swap_polish(a, b, best_p, n, t0)
    if c2 < best_cost:
        best_p = p2
        best_cost = c2

    # 3) If time remains, do a second polish pass from the improved solution
    if time.time() < t0 - 1.0:
        p3, c3 = _swap_polish(a, b, best_p, n, t0)
        if c3 < best_cost:
            best_p = p3
            best_cost = c3

    return best_p.tolist()
