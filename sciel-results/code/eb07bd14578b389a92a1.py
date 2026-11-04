"""QAP tai256c solver: rank-matching seed + fast 2-swap local search with restarts."""
import time
import numpy as np


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """Cost change from swapping locations of facilities i and j (p[i]<->p[j])."""
    # cost = sum_{u,v} a[u,v] * b[p[u],p[v]]
    # Swapping p[i] and p[j] changes terms involving i or j.
    # Let x = p[i], y = p[j]. After swap p[i]=y, p[j]=x.
    # Affected terms: (i,*) , (*,i), (j,*) , (*,j) but (i,j),(j,i) counted.
    # delta = sum over affected of a * (new_b - old_b)
    # Efficient: delta = 2 * [ sum_{v != i,j} a[i,v]*(b[y,p[v]]-b[x,p[v]])
    #                       + sum_{v != i,j} a[j,v]*(b[x,p[v]]-b[y,p[v]])
    #                       + a[i,j]*(b[x,y]-b[y,x]) ]  ... but need care.
    # Simpler exact form:
    # Let A_i = a[i,:], A_j = a[j,:]
    # old contribution of row i: A_i . b[x, p]
    # new: A_i . b[y, p]
    # old row j: A_j . b[y, p]; new: A_j . b[x, p]
    # But b[x,p] includes b[x, p[i]]=b[x,x] and b[x,p[j]]=b[x,y]; etc.
    # delta = A_i . (b[y,p]-b[x,p]) + A_j . (b[x,p]-b[y,p])
    #       = (A_i - A_j) . (b[y,p] - b[x,p])
    x = p[i]
    y = p[j]
    # b[y,p] - b[x,p] is a vector; (a[i,:]-a[j,:]) dotted with it
    diff_b = b[y, p] - b[x, p]
    diff_a = a[i, :] - a[j, :]
    return int(np.dot(diff_a, diff_b))


def solve(a, b):
    n = len(a)
    deadline = time.time() + 140.0  # safety margin under 150s kill

    # ---- initial solution: rank-matching (heavy flow -> short distance) ----
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    cur = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = cur

    # Precompute row sums for a and b (used in faster delta approximations if needed)
    # We'll do a full 2-swap local search with multiple passes / restarts.

    def local_search(p, cur, max_iter=None):
        nonlocal deadline
        improved = True
        it = 0
        # Precompute a and b as float for speed? keep int64.
        # Use a pass-based 2-opt: scan all pairs, apply improving swaps.
        while improved and it < 200:
            improved = False
            it += 1
            # randomize order each pass to escape plateaus
            order = np.random.permutation(n)
            for ii in range(n):
                if time.time() > deadline:
                    return p, cur
                i = order[ii]
                # For each j != i, compute delta
                ai = a[i, :]
                pi = p[i]
                for jj in range(ii + 1, n):
                    j = order[jj]
                    if i == j:
                        continue
                    d = _swap_delta(a, b, p, i, j)
                    if d < 0:
                        # accept swap
                        p[i], p[j] = p[j], p[i]
                        cur += d
                        improved = True
        return p, cur

    # First local search pass
    p, cur = local_search(p, cur)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()

    # ---- restarts with perturbation while time remains ----
    rng = np.random.default_rng(0)
    restart = 0
    while time.time() < deadline - 5.0:
        restart += 1
        # perturb best solution: random k swaps
        p = best_p.copy()
        cur = best_cost
        k = min(n, 4 + restart)
        for _ in range(k):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
                cur += _swap_delta(a, b, p, i, j)
        # local search from perturbed
        p, cur = local_search(p, cur)
        if cur < best_cost:
            best_cost = cur
            best_p = p.copy()

    return best_p.tolist()
