"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _qap_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    b_p = b[np.ix_(p, p)]
    return int(np.sum(a * b_p))

def _init_rank_matching(a, b):
    """Rank-matching initialization: sort facilities by flow centrality, locations by distance centrality."""
    n = len(a)
    flow_centrality = a.sum(axis=0) + a.sum(axis=1)
    dist_centrality = b.sum(axis=0) + b.sum(axis=1)
    
    heavy = np.argsort(-flow_centrality, kind="stable")
    central = np.argsort(dist_centrality, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _init_greedy(a, b):
    """Greedy construction: place facilities one by one in order of flow centrality."""
    n = len(a)
    flow_centrality = a.sum(axis=0) + a.sum(axis=1)
    order = np.argsort(-flow_centrality, kind="stable")
    
    p = np.full(n, -1, dtype=np.int64)
    used_locs = np.zeros(n, dtype=bool)
    
    for fac in order:
        best_loc = -1
        best_cost = np.inf
        for loc in range(n):
            if used_locs[loc]:
                continue
            placed_mask = p >= 0
            if not np.any(placed_mask):
                cost = 0
            else:
                placed_facs = np.where(placed_mask)[0]
                cost = np.dot(a[fac, placed_facs], b[loc, p[placed_facs]]) + \
                       np.dot(a[placed_facs, fac], b[p[placed_facs], loc])
            if cost < best_cost:
                best_cost = cost
                best_loc = loc
        p[fac] = best_loc
        used_locs[best_loc] = True
    
    return p

def _swap_delta(a, b, p, i, j):
    """
    Compute cost delta for swapping p[i] and p[j].
    
    Cost = sum_{i,j} a[i,j]*b[p[i],p[j]]
    
    When we swap p[i]<->p[j], the affected terms are those where index i or j appears
    as either the row or column index in the sum.
    
    Let li = p[i], lj = p[j].
    
    Affected terms:
    1. Terms a[i,k]*b[li,p[k]] for all k -> a[i,k]*b[lj,p[k]] for k!=i,j
       Change: sum_{k!=i,j} a[i,k]*(b[lj,p[k]]-b[li,p[k]])
    2. Terms a[k,i]*b[p[k],li] for all k -> a[k,i]*b[p[k],lj] for k!=i,j
       Change: sum_{k!=i,j} a[k,i]*(b[p[k],lj]-b[p[k],li])
    3. Terms a[j,k]*b[lj,p[k]] for all k -> a[j,k]*b[li,p[k]] for k!=i,j
       Change: sum_{k!=i,j} a[j,k]*(b[li,p[k]]-b[lj,p[k]])
    4. Terms a[k,j]*b[p[k],lj] for all k -> a[k,j]*b[p[k],li] for k!=i,j
       Change: sum_{k!=i,j} a[k,j]*(b[p[k],li]-b[p[k],lj])
    5. Cross terms (i,j) and (j,i):
       a[i,j]*b[li,lj] -> a[i,j]*b[lj,li]
       a[j,i]*b[lj,li] -> a[j,i]*b[li,lj]
       Change: a[i,j]*(b[lj,li]-b[li,lj]) + a[j,i]*(b[li,lj]-b[lj,li])
    
    Note: terms a[i,i]*b[li,li] and a[j,j]*b[lj,lj] become a[i,i]*b[lj,lj] and a[j,j]*b[li,li]
       Change: a[i,i]*(b[lj,lj]-b[li,li]) + a[j,j]*(b[li,li]-b[lj,lj])
    """
    li = p[i]
    lj = p[j]
    
    # Create mask excluding i and j
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    
    # Term 1: sum_{k!=i,j} a[i,k]*(b[lj,p[k]]-b[li,p[k]])
    t1 = np.dot(a[i, mask], b[lj, p[mask]] - b[li, p[mask]])
    
    # Term 2: sum_{k!=i,j} a[k,i]*(b[p[k],lj]-b[p[k],li])
    t2 = np.dot(a[mask, i], b[p[mask], lj] - b[p[mask], li])
    
    # Term 3: sum_{k!=i,j} a[j,k]*(b[li,p[k]]-b[lj,p[k]])
    t3 = np.dot(a[j, mask], b[li, p[mask]] - b[lj, p[mask]])
    
    # Term 4: sum_{k!=i,j} a[k,j]*(b[p[k],li]-b[p[k],lj])
    t4 = np.dot(a[mask, j], b[p[mask], li] - b[p[mask], lj])
    
    # Cross terms
    t5 = a[i, j] * (b[lj, li] - b[li, lj]) + a[j, i] * (b[li, lj] - b[lj, li])
    
    # Diagonal terms
    t6 = a[i, i] * (b[lj, lj] - b[li, li]) + a[j, j] * (b[li, li] - b[lj, lj])
    
    return int(t1 + t2 + t3 + t4 + t5 + t6)

def _sa_run(a, b, p_init, max_time, rng, t_start, t_end, cooling_rate):
    """Run simulated annealing for max_time seconds, return best permutation and cost."""
    n = len(a)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    steps = 0
    accepted = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
        if remaining := (max_time - elapsed) < 0.05:
            break
        
        # Process in small batches with time check
        batch = 2000
        for _ in range(batch):
            if time.time() - start_time > max_time:
                break
            
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            
            delta = _swap_delta(a, b, p, i, j)
            
            if delta <= 0:
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                accepted += 1
            else:
                if t_start > t_end:
                    # Accept with probability exp(-delta/t)
                    # Use log for numerical stability
                    if delta / t_start < 700:
                        if math.log(rng.random()) < -delta / t_start:
                            p[i], p[j] = p[j], p[i]
                            current_cost += delta
                            accepted += 1
            
            # Cool
            t_start *= cooling_rate
            if t_start < t_end:
                t_start = t_end
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        steps += batch
    
    return best_p, best_cost

def _hill_climb(a, b, p, max_time):
    """First-improvement hill climbing with 2-opt swaps."""
    n = len(a)
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    start = time.time()
    
    improved = True
    while improved and time.time() - start < max_time:
        improved = False
        for i in range(n):
            if time.time() - start > max_time:
                break
            for j in range(i + 1, n):
                if time.time() - start > max_time:
                    break
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    max_total_time = 115.0
    
    rng = np.random.default_rng(42)
    
    best_p = None
    best_cost = np.inf
    
    # Run 1: Rank matching + long SA
    p_init = _init_rank_matching(a, b)
    p_result, cost_result = _sa_run(a, b, p_init, 38.0, rng, 50000.0, 1.0, 0.99992)
    if cost_result < best_cost:
        best_cost = cost_result
        best_p = p_result
    
    # Run 2: Perturb best, shorter SA with higher temp
    if time.time() - start_time < max_total_time - 40.0:
        p_init = best_p.copy()
        n_swaps = n // 8
        for _ in range(n_swaps):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            p_init[i], p_init[j] = p_init[j], p_init[i]
        p_result, cost_result = _sa_run(a, b, p_init, 35.0, rng, 20000.0, 1.0, 0.99995)
        if cost_result < best_cost:
            best_cost = cost_result
            best_p = p_result
    
    # Run 3: Greedy init + SA
    if time.time() - start_time < max_total_time - 30.0:
        p_init = _init_greedy(a, b)
        p_result, cost_result = _sa_run(a, b, p_init, 30.0, rng, 30000.0, 1.0, 0.99993)
        if cost_result < best_cost:
            best_cost = cost_result
            best_p = p_result
    
    # Final hill climb
    if best_p is not None and time.time() - start_time < max_total_time - 1.0:
        remaining = max_total_time - 1.0 - (time.time() - start_time)
        p_result, cost_result = _hill_climb(a, b, best_p, remaining)
        if cost_result < best_cost:
            best_cost = cost_result
            best_p = p_result
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
