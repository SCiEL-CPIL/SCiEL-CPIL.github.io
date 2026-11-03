"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    pi = p[i]
    pj = p[j]
    bpj = b[:, pj][p]
    bpi = b[:, pi][p]
    brj = b[pj, :][p]
    bri = b[pi, :][p]
    delta = (a[:, i] * (bpj - bpi) +
             a[:, j] * (bpi - bpj) +
             a[i, :] * (brj - bri) +
             a[j, :] * (bri - brj))
    delta[i] = 0
    delta[j] = 0
    direct = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
    return int(np.sum(delta) + direct)


def _local_search(a, b, p, max_passes=5, time_limit=5.0):
    """Steepest descent local search: repeatedly find the best improving swap."""
    n = len(p)
    start = time.time()
    for _ in range(max_passes):
        if time.time() - start > time_limit:
            break
        best_d = 0
        best_i, best_j = -1, -1
        for i in range(n):
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < best_d:
                    best_d = d
                    best_i, best_j = i, j
        if best_i == -1:
            break
        p[best_i], p[best_j] = p[best_j], p[best_i]
    return p


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Initial local search to polish the rank-matching start
    p = _local_search(a, b, p, max_passes=3, time_limit=5.0)
    current_cost = _qap_cost(a, b, p)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    T = 5000.0
    T_min = 1.0
    cooling = 0.9999
    rng = np.random.default_rng(42)
    sa_count = 0
    while time.time() < deadline:
        if T < T_min:
            # Periodically do local search on best solution
            if sa_count % 50000 == 0:
                lp = best_p.copy()
                lp = _local_search(a, b, lp, max_passes=2, time_limit=3.0)
                lc = _qap_cost(a, b, lp)
                if lc < best_cost:
                    best_cost = lc
                    best_p = lp.copy()
            T = 5000.0
            p = best_p.copy()
            current_cost = best_cost
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        d = _swap_delta(a, b, p, i, j)
        if d < 0 or rng.random() < np.exp(-d / T):
            p[i], p[j] = p[j], p[i]
            current_cost += d
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        T *= cooling
        sa_count += 1
    return best_p.tolist()
# EVOLVE-BLOCK-END
