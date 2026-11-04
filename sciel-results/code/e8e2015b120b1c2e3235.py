"""
QAP solver for wil100: multi-start vectorized steepest-descent local search
with a final simulated-annealing polish.  Kept inside the same design family
as the previous program (greedy/Hungarian/random starts, O(n^2) delta matrix,
SA polish) but with tighter time budgeting and a few small structural changes.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Exact integer QAP cost."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _greedy_initial(a, b):
    n = a.shape[0]
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    heavy = np.argsort(-flow_sums, kind="stable")
    central = np.argsort(dist_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _hungarian_initial(a, b):
    n = a.shape[0]
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    cost = np.outer(flow_sums, dist_sums)
    row_ind, col_ind = linear_sum_assignment(cost)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _random_initial(n):
    return np.random.permutation(n).astype(np.int64)


def _compute_delta_matrix(a, b, p):
    """Vectorised O(n^2) swap-delta matrix."""
    n = len(p)
    Bp = b[np.ix_(p, p)]
    M1 = a.T @ Bp
    M2 = a @ Bp
    d1 = np.diag(M1)
    d2 = np.diag(M2)
    dB = np.diag(Bp)
    aD = np.diag(a)

    T1 = M1 - d1[:, None] - d1[None, :] + M1.T
    T2 = M2 - d2[:, None] - d2[None, :] + M2.T

    C1p1 = (aD[:, None] - a) * (Bp - dB[:, None])
    C1p2 = (a.T - aD[None, :]) * (dB[None, :] - Bp)
    C1 = C1p1 + C1p2

    C2p1 = (aD[:, None] - a.T) * (Bp.T - dB[:, None])
    C2p2 = (a - aD[None, :]) * (dB[None, :] - Bp.T)
    C2 = C2p1 + C2p2

    T3 = (a - a.T) * (Bp.T - Bp)

    D = (T1 - C1) + (T2 - C2) + T3
    np.fill_diagonal(D, 0)
    return D


def _steepest_local_search(a, b, p, time_limit, start_time):
    """Steepest-descent 2-swap local search until no improving move."""
    n = len(p)
    while time.time() - start_time < time_limit:
        D = _compute_delta_matrix(a, b, p)
        best = int(np.min(D))
        if best >= 0:
            break
        i, j = np.unravel_index(np.argmin(D), D.shape)
        p[i], p[j] = p[j], p[i]
    return p


def _swap_delta(a, b, p, i, j):
    pi, pj = p[i], p[j]
    T1 = np.dot(a[:, i] - a[:, j], b[p, pj] - b[p, pi])
    T1 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    T1 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])

    T2 = np.dot(a[i, :] - a[j, :], b[pj, p] - b[pi, p])
    T2 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    T2 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])

    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    return int(T1 + T2 + T3)


def solve(a, b):
    n = len(a)
    deadline = time.time() + 112.0
    t0 = time.time()

    best_p = None
    best_cost = float("inf")

    # Build a pool of diverse starting points
    starts = [_greedy_initial(a, b), _hungarian_initial(a, b)]
    for _ in range(4):
        starts.append(_random_initial(n))

    # Give each start a fair slice of the budget; keep a reserve for SA
    reserve = 30.0
    per_start = max(2.0, (112.0 - reserve) / len(starts))

    for s in starts:
        if time.time() >= deadline - reserve - 5.0:
            break
        p = s.copy()
        cost = _qap_cost(a, b, p)
        limit = min(time.time() + per_start, deadline - reserve - 2.0)
        p = _steepest_local_search(a, b, p, limit, t0)
        cost = _qap_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    if best_p is None:
        best_p = _greedy_initial(a, b)
        best_cost = _qap_cost(a, b, best_p)

    # SA polish on the best solution
    p = best_p.copy()
    cost = best_cost
    T_hi = max(1.0, best_cost * 0.005)
    T_lo = 0.01

    # Precompute row/column sums for a quick temperature schedule
    n_steps_estimate = max(1, int((deadline - time.time()) * 8))
    step = 0

    while time.time() < deadline - 0.5:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        d = _swap_delta(a, b, p, i, j)
        frac = min(1.0, (step / max(1, n_steps_estimate)))
        T = T_hi * (T_lo / T_hi) ** frac
        step += 1
        if d < 0 or np.random.rand() < np.exp(-d / T):
            p[i], p[j] = p[j], p[i]
            cost += d
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
