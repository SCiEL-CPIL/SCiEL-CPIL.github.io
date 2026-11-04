# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _delta_swap(a, b, p, i, j):
    pi = p[i]
    pj = p[j]
    n = len(p)
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    diff1 = b[pj, p[mask]] - b[pi, p[mask]]
    diff2 = b[p[mask], pj] - b[p[mask], pi]
    a_diff_row = a[i, mask] - a[j, mask]
    a_diff_col = a[mask, i] - a[mask, j]
    delta_k = np.sum(a_diff_row * diff1) + np.sum(a_diff_col * diff2)
    delta_self = (a[i, j] * (b[pj, pi] - b[pi, pj])
                  + a[j, i] * (b[pi, pj] - b[pj, pi])
                  + a[i, i] * (b[pj, pj] - b[pi, pi])
                  + a[j, j] * (b[pi, pi] - b[pj, pj]))
    return int(delta_k + delta_self)


def _sinkhorn_relax(a, b, n, rng, deadline, n_iter=200):
    """
    Entropic (Sinkhorn-regularized) alternating minimization on the Birkhoff
    polytope. Objective: f(X) = sum_{i,j} a[i,j] * b[X] = <B, X^T A X>
    We minimize using alternating coordinate updates: fix X, the gradient wrt
    X is 2 * A X B^T + 2 * A^T X B (for general A,B). We do a multiplicative
    (Sinkhorn) update driven by the gradient, which keeps X row/column
    normalized and is equivalent to an entropic-regularized relaxation.
    """
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0

    # Initialize from a random doubly stochastic matrix
    X = np.ones((n, n), dtype=np.float64) / n

    # Anneal the regularization: start with strong smoothing, reduce over time
    # We use a temperature-like step scale that decays
    for it in range(n_iter):
        if time.time() > deadline:
            break
        # Gradient of <B, X^T A X> w.r.t X: (A + A^T) X B + (A^T + A) X B^T
        # For our use, use symmetric parts to keep it stable
        grad = 2.0 * (a_sym @ X @ b_sym)

        # Multiplicative update (Sinkhorn-style): X <- X * exp(-eta * grad)
        # Scale eta so the update is stable
        eta = 1.0 / (max(1.0, np.abs(grad).max()) * 2.0)
        # Decay eta over iterations to sharpen
        eta *= (1.0 - 0.5 * it / max(1, n_iter))

        logX = np.log(X + 1e-300)
        logX = logX - eta * grad
        # Normalize rows
        row_max = logX.max(axis=1, keepdims=True)
        X = np.exp(logX - row_max)
        X /= X.sum(axis=1, keepdims=True)
        # Normalize columns
        col_max = X.sum(axis=0, keepdims=True)
        col_max[col_max == 0] = 1
        X /= col_max
        # Renormalize rows again (Sinkhorn alternating)
        for _ in range(3):
            rs = X.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1
            X /= rs
            cs = X.sum(axis=0, keepdims=True)
            cs[cs == 0] = 1
            X /= cs

    # Extract permutation via Hungarian on -X
    row_ind, col_ind = linear_sum_assignment(-X)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _sinkhorn_from_perm(a, b, p, deadline, n_iter=100):
    """
    Start from the permutation matrix and refine via Sinkhorn-regularized
    alternating minimization, then re-extract a (possibly improved) permutation.
    """
    n = len(p)
    X = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        X[i, p[i]] = 1.0
    # Add small entropy to allow movement
    X = X * 0.9 + 0.1 / n

    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0

    for it in range(n_iter):
        if time.time() > deadline:
            break
        grad = 2.0 * (a_sym @ X @ b_sym)
        eta = 0.5 / (max(1.0, np.abs(grad).max()) * 2.0)
        eta *= (1.0 - 0.5 * it / max(1, n_iter))
        logX = np.log(X + 1e-300)
        logX = logX - eta * grad
        row_max = logX.max(axis=1, keepdims=True)
        X = np.exp(logX - row_max)
        X /= X.sum(axis=1, keepdims=True)
        for _ in range(3):
            rs = X.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1
            X /= rs
            cs = X.sum(axis=0, keepdims=True)
            cs[cs == 0] = 1
            X /= cs

    row_ind, col_ind = linear_sum_assignment(-X)
    p_new = np.empty(n, dtype=np.int64)
    p_new[row_ind] = col_ind
    return p_new


def _vectorized_best_improvement(a, b, p, deadline, max_rounds=50):
    """
    Vectorized best-improvement 2-swap local search.
    Each round: compute all swap deltas, apply the best improving swap.
    """
    n = len(p)
    cost = _cost(a, b, p)
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        # Precompute row/col contributions
        # delta(i,j) = sum_{k!=i,j} [a[i,k](b[pj, pk]-b[pi, pk]) + a[k,i](b[pk, pj]-b[pk, pi])
        #               + a[j,k](b[pi, pk]-b[pj, pk]) + a[k,j](b[pk, pi]-b[pk, pj])
        #             + self terms
        # Vectorize over pairs by building matrices
        best_delta = 0
        best_i = -1
        best_j = -1
        # For each i, compute deltas with all j > i
        for i in range(n):
            if time.time() > deadline:
                break
            pi = p[i]
            # deltas for j in i+1..n-1
            # diff1[j] = b[pj, p[k]] - b[pi, p[k]] for all k != i,j
            # We'll compute per j but vectorize the inner sum over k
            js = np.arange(i + 1, n)
            if len(js) == 0:
                continue
            pjs = p[js]
            # For each j, mask excludes i and j
            # a[i, k] for k != i, j  -> a[i, :] with i removed
            # Build diff1 matrix: shape (len(js), n) where diff1[m, k] = b[pjs[m], p[k]] - b[pi, p[k]]
            diff1 = b[np.ix_(pjs, p)] - b[pi, p]  # (len(js), n)
            diff2 = b[np.ix_(p, pjs)] - b[p, pi]  # (n, len(js))
            # a[i, k] for k != i: a[i, :]
            a_i = a[i, :].copy()
            a_i[i] = 0  # exclude k=i
            # a[k, i] for k != i
            a_ki = a[:, i].copy()
            a_ki[i] = 0
            # For each j (index m in js), exclude k = j
            # delta_k[m] = sum_k a_i[k] * diff1[m,k] + sum_k a_ki[k] * diff2[k,m]
            delta_k = a_i @ diff1.T  # (len(js),)
            delta_k += diff2.T @ a_ki  # (len(js),)
            # Now subtract the k=j term (since k=j should be excluded from the sum)
            # For m-th j (actual index j=js[m]), k=j term:
            # a_i[j] * diff1[m, j] + a_ki[j] * diff2[j, m]
            j_idx = js
            delta_k -= a_i[j_idx] * diff1[np.arange(len(js)), j_idx]
            delta_k -= a_ki[j_idx] * diff2[j_idx, np.arange(len(js))]
            # Self terms
            # a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
            # + a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
            self_terms = (a[i, j_idx] * (b[pjs, pi] - b[pi, pjs])
                          + a[j_idx, i] * (b[pi, pjs] - b[pjs, pi])
                          + a[i, i] * (b[np.ix_(pjs, pjs)] - b[pi, pi])
                          + a[j_idx, j_idx] * (b[pi, pi] - b[np.ix_(pjs, pjs)]))
            deltas = delta_k + self_terms
            # Find best in this row
            local_best = int(np.argmin(deltas))
            if deltas[local_best] < best_delta:
                best_delta = deltas[local_best]
                best_i = i
                best_j = js[local_best]
        if best_i >= 0 and best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta
        else:
            break
    return p, cost


def _first_improvement_ls(a, b, p, deadline, max_passes=20):
    n = len(p)
    cost = _cost(a, b, p)
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p, cost
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    break
        if not improved:
            break
    return p, cost


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 118.0  # safety margin before 150s kill

    best_p = None
    best_cost = np.inf
    rng = np.random.default_rng(2024)

    # Phase 1: Sinkhorn-entropic relaxation from a diffuse start (multiple restarts)
    n_restarts = 3
    per_restart = (deadline - 30.0) / n_restarts
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        sub_deadline = min(deadline - 30.0, t_start + (r + 1) * per_restart)
        p = _sinkhorn_relax(a, b, n, rng, sub_deadline, n_iter=200)
        p, c = _vectorized_best_improvement(a, b, p, deadline - 25.0)
        if c < best_cost:
            best_cost = c
            best_p = p.copy()

    # Phase 2: Sinkhorn refinement from the best permutation (escape local minima)
    if time.time() < deadline - 20.0:
        p_ref = _sinkhorn_from_perm(a, b, best_p, deadline - 15.0, n_iter=80)
        p_ref, c_ref = _vectorized_best_improvement(a, b, p_ref, deadline - 10.0)
        if c_ref < best_cost:
            best_cost = c_ref
            best_p = p_ref.copy()

    # Phase 3: A couple of Sinkhorn-from-best iterations to escape
    for _ in range(2):
        if time.time() > deadline - 10.0:
            break
        p_ref = _sinkhorn_from_perm(a, b, best_p, deadline - 8.0, n_iter=60)
        p_ref, c_ref = _vectorized_best_improvement(a, b, p_ref, deadline - 5.0)
        if c_ref < best_cost:
            best_cost = c_ref
            best_p = p_ref.copy()

    # Phase 4: Final polish with vectorized best-improvement
    if time.time() < deadline - 3.0:
        p_pol, c_pol = _vectorized_best_improvement(a, b, best_p.copy(), deadline - 1.0, max_rounds=100)
        if c_pol < best_cost:
            best_cost = c_pol
            best_p = p_pol.copy()

    # Phase 5: One first-improvement pass for any remaining easy gains
    if time.time() < deadline - 0.5:
        p_fin, c_fin = _first_improvement_ls(a, b, best_p.copy(), deadline - 0.1, max_passes=5)
        if c_fin < best_cost:
            best_cost = c_fin
            best_p = p_fin.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
