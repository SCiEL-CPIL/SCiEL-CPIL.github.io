"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute the QAP cost for permutation p using vectorized ops."""
    p_arr = np.asarray(p, dtype=np.int64)
    # b_perm[i,j] = b[p[i], p[j]]
    b_perm = b[np.ix_(p_arr, p_arr)]
    return int(np.sum(a * b_perm))


def _greedy_initial(a, b, n):
    """Create a good initial solution using a vectorized greedy heuristic."""
    # Precompute
    a_row_sums = a.sum(axis=1)
    
    # Sort facilities by importance (row sum of a) in descending order
    facility_order = np.argsort(-a_row_sums, kind='stable')
    
    assigned = np.full(n, -1, dtype=np.int64)
    used_mask = np.zeros(n, dtype=bool)
    
    for idx in facility_order:
        # For each unused location, compute incremental cost
        unused_locs = np.where(~used_mask)[0]
        
        # Incremental cost for assigning facility idx to location loc:
        # sum over already-assigned k: a[idx,k]*b[loc,assigned[k]] + a[k,idx]*b[assigned[k],loc]
        # + a[idx,idx]*b[loc,loc]
        
        inc_costs = np.zeros(len(unused_locs))
        for t, loc in enumerate(unused_locs):
            # Self term
            inc_costs[t] = a[idx, idx] * b[loc, loc]
            # Cross terms with already assigned
            assigned_mask = assigned >= 0
            if assigned_mask.any():
                a_k = a[idx, assigned_mask]
                b_loc_k = b[loc, assigned[assigned_mask]]
                inc_costs[t] += np.dot(a_k, b_loc_k)
                a_k2 = a[assigned_mask, idx]
                b_k_loc = b[assigned[assigned_mask], loc]
                inc_costs[t] += np.dot(a_k2, b_k_loc)
        
        best_t = np.argmin(inc_costs)
        best_loc = unused_locs[best_t]
        assigned[idx] = best_loc
        used_mask[best_loc] = True
    
    return assigned.tolist()


def _vectorized_delta(a, b, p, i, j):
    """Compute delta cost for swapping p[i] and p[j] using vectorized ops."""
    n = len(p)
    p_arr = np.asarray(p, dtype=np.int64)
    li = p_arr[i]
    lj = p_arr[j]
    
    # Create mask for k not in {i, j}
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    p_k = p_arr[mask]
    
    # Vectorized delta computation
    # (i,i) + (j,j)
    delta = a[i, i] * (b[lj, lj] - b[li, li])
    delta += a[j, j] * (b[li, li] - b[lj, lj])
    # (i,j) + (j,i)
    delta += a[i, j] * (b[lj, li] - b[li, lj])
    delta += a[j, i] * (b[li, lj] - b[lj, li])
    
    # (i,k) terms: a[i,k] * (b[lj, p[k]] - b[li, p[k]])
    delta += np.dot(a[i, mask], b[lj, p_k] - b[li, p_k])
    # (j,k) terms: a[j,k] * (b[li, p[k]] - b[lj, p[k]])
    delta += np.dot(a[j, mask], b[li, p_k] - b[lj, p_k])
    # (k,i) terms: a[k,i] * (b[p[k], lj] - b[p[k], li])
    delta += np.dot(a[mask, i], b[p_k, lj] - b[p_k, li])
    # (k,j) terms: a[k,j] * (b[p[k], li] - b[p[k], lj])
    delta += np.dot(a[mask, j], b[p_k, li] - b[p_k, lj])
    
    return int(delta)


def _local_search_2opt(a, b, p, time_limit=100):
    """Improve the solution using vectorized 2-opt local search."""
    p = list(p)
    n = len(p)
    start_time = time.time()
    current_cost = _qap_cost(a, b, p)
    
    best_cost = current_cost
    best_p = p[:]
    
    rng = np.random.RandomState(42)
    
    max_iterations = 200000
    iteration = 0
    no_improve_count = 0
    
    while iteration < max_iterations:
        if time.time() - start_time > time_limit:
            break
        
        # Pick two random distinct facilities
        i, j = rng.choice(n, size=2, replace=False)
        
        delta = _vectorized_delta(a, b, p, int(i), int(j))
        new_cost = current_cost + delta
        
        if new_cost < current_cost:
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
            no_improve_count = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p[:]
        else:
            no_improve_count += 1
        
        iteration += 1
        
        # Early termination if stuck for a long time
        if no_improve_count > 50000:
            break
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    
    # Get a good initial solution using greedy
    p = _greedy_initial(a, b, n)
    
    # Run local search to improve
    p, cost = _local_search_2opt(a, b, p, time_limit=100)
    
    return p
# EVOLVE-BLOCK-END
