# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """Compute exact delta for swapping facilities at positions i and j.
    delta = sum_k (a[i,k] - a[k,j]) * (b[p[j],k] - b[p[i],k])
    Note: b[p[j],k] means b[p[j], p[k]] for all k, but we need b at the mapped positions.
    Actually: after swap, facility i goes to p[j] and facility j goes to p[i].
    delta = sum_k a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + sum_k a[k,j]*(b[p[i],p[k]] - b[p[j],p[k]])
          = sum_k (a[i,k] - a[k,j]) * (b[p[j],p[k]] - b[p[i],p[k]])
    """
    pi = p[i]
    pj = p[j]
    # dB[k] = b[pj, p[k]] - b[pi, p[k]] for all k
    dB = b[pj, p] - b[pi, p]
    # dA[k] = a[i,k] - a[k,j]
    dA = a[i, :] - a[:, j]
    return int(np.dot(dA, dB))


def _refine(a, b, p, deadline, best_cost, best_p):
    """First-improvement 2-opt (swap) local search until deadline.
    p is a numpy int64 array (a copy).
    Returns (best_cost, best_p) found so far."""
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
                    
                    # Time check
                    if time.time() >= deadline:
                        return best_cost, best_p
        if not improved:
            break

    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    return best_cost, best_p


def _anneal(a, b, p, start_time, deadline, best_cost, best_p, rng):
    """Simulated Annealing refinement.
    Returns (best_cost, best_p)."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()

    T = 5000.0
    T_end = 1.0
    alpha = 0.995
    
    while T > T_end and (time.time() - start_time) < deadline:
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
            
        delta = _swap_delta(a, b, p, i, j)
        
        # Acceptance criterion
        if delta < 0 or (rng.random() < np.exp(-delta / T)):
            p[i], p[j] = p[j], p[i]
            cur += delta
            
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
                
        T *= alpha
        
        # Periodic time check
        if time.time() - start_time > deadline:
            break
            
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
        
    return best_cost, best_p


def _benders_surrogate(a, b, p):
    """Compute a Benders-style surrogate cost for each facility-location pair.
    For facility i at location l, the surrogate is sum_j a[i,j] * b[l, p[j]]
    This gives a linearized view of the cost contribution.
    """
    n = len(p)
    B = b[np.ix_(p, p)]
    surrogate = a @ B  # surrogate[i] = sum_j a[i,j] * b[p[i], p[j]]
    return surrogate


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s kill

    # --- Strong constructive start using Benders-style surrogate ---
    # Compute initial surrogate to guide assignment
    # First, use rank matching as a starting point
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    
    best_cost = _qap_cost(a, b, p0)
    best_p = p0.copy()

    # First refinement on the rank-matching start
    p = p0.copy()
    best_cost, best_p = _refine(a, b, p, deadline, best_cost, best_p)

    # --- Simulated Annealing for diversification ---
    rng = np.random.default_rng(12345)
    
    # We can run SA on the current best solution found by local search
    p = best_p.copy()
    best_cost, best_p = _anneal(a, b, p, t0, deadline, best_cost, best_p, rng)
    
    # If time remains, do a few random restarts with local search
    while time.time() < deadline - 5.0:
        p = rng.permutation(n).astype(np.int64)
        best_cost, best_p = _refine(a, b, p, deadline, best_cost, best_p)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
