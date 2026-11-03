# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """Compute the exact cost change when swapping locations p[i] and p[j]."""
    pi, pj = p[i], p[j]
    if i == j:
        return 0
    
    dA = a[i, :] - a[j, :]
    dB = b[pj, p] - b[pi, p]
    
    full_dot = np.dot(dA, dB)
    
    term_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    term_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    sum_others = full_dot - term_i - term_j
    
    delta_ij = a[i, i] * (b[pj, pi] - b[pi, pi]) \
             + a[i, j] * (b[pj, pj] - b[pi, pj]) \
             + a[j, i] * (b[pi, pi] - b[pj, pi]) \
             + a[j, j] * (b[pi, pj] - b[pj, pj])
    
    return int(sum_others + delta_ij)


def _refine(a, b, p, deadline, best_cost, best_p):
    """First-improvement 2-opt (swap) local search until deadline."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()

    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cur += delta
                    improved = True
                    
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
        if not improved:
            break

    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    return best_cost, best_p


def _anneal(a, b, p, start_time, deadline, best_cost, best_p, rng):
    """Simulated Annealing refinement."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()

    T = 5000.0
    T_end = 1.0
    alpha = 0.999
    
    steps = 0
    while T > T_end and (time.time() - start_time) < deadline:
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
            
        delta = _swap_delta(a, b, p, i, j)
        
        if delta < 0 or (rng.random() < np.exp(-delta / T)):
            p[i], p[j] = p[j], p[i]
            cur += delta
            
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
                
        T *= alpha
        steps += 1
        
        if steps % 50 == 0 and time.time() - start_time > deadline:
            break
            
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
        
    return best_cost, best_p


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 118.0

    rng = np.random.default_rng(42)
    
    best_cost = np.iinfo(np.int64).max
    best_p = None

    # Strategy 1: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    
    p = p0.copy()
    cur_cost = _qap_cost(a, b, p)
    best_cost, best_p = cur_cost, p.copy()
    
    p = p0.copy()
    best_cost, best_p = _refine(a, b, p, deadline, best_cost, best_p)
    
    # Strategy 2: Reverse rank matching
    if time.time() < deadline - 30:
        heavy2 = np.argsort(a.sum(axis=1), kind="stable")
        central2 = np.argsort(-b.sum(axis=1), kind="stable")
        p1 = np.empty(n, dtype=np.int64)
        p1[heavy2] = central2
        p = p1.copy()
        best_cost, best_p = _refine(a, b, p, deadline, best_cost, best_p)
    
    # Simulated Annealing from best
    if time.time() < deadline - 15:
        p = best_p.copy()
        best_cost, best_p = _anneal(a, b, p, t0, deadline, best_cost, best_p, rng)
    
    # Random restarts with local search
    restarts = 0
    while time.time() < deadline - 5.0:
        p = rng.permutation(n).astype(np.int64)
        best_cost, best_p = _refine(a, b, p, deadline, best_cost, best_p)
        restarts += 1
        if restarts >= 3:
            break
    
    # Final polish
    if time.time() < deadline - 1.0:
        p = best_p.copy()
        best_cost, best_p = _refine(a, b, p, deadline, best_cost, best_p)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
