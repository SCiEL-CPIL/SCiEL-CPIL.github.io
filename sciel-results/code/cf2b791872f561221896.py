"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in QAP cost when swapping facilities at positions i and j.
    Uses vectorized dot products for speed.
    """
    u = p[i]
    v = p[j]
    
    # Vectorized sums over all k
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[v, p] - b[u, p]
    S1 = np.dot(diff_a_row, diff_b_col)
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, v] - b[p, u]
    S2 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract terms for k=i and k=j from S1 and S2
    # Term k=i in S1: (a[i,i] - a[j,i]) * (b[v, p[i]] - b[u, p[i]]) = (a_ii - a_ji) * (b_vu - b_uu)
    # Term k=j in S1: (a[i,j] - a[j,j]) * (b[v, p[j]] - b[u, p[j]]) = (a_ij - a_jj) * (b_vv - b_uv)
    
    # Term k=i in S2: (a[i,i] - a[i,j]) * (b[p[i], v] - b[p[i], u]) = (a_ii - a_ij) * (b_uv - b_uu)
    # Term k=j in S2: (a[j,i] - a[j,j]) * (b[p[j], v] - b[p[j], u]) = (a_ji - a_jj) * (b_vv - b_vu)
    
    err1_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
    err1_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
    
    err2_i = (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
    err2_j = (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
    
    # Delta for cross terms (k != i, j)
    delta_cross = S1 + S2 - (err1_i + err1_j + err2_i + err2_j)
    
    # Delta for self and pair terms (i, i), (j, j), (i, j), (j, i)
    old_self = a[i, i] * b[u, u] + a[j, j] * b[v, v] + a[i, j] * b[u, v] + a[j, i] * b[v, u]
    new_self = a[i, i] * b[v, v] + a[j, j] * b[u, u] + a[i, j] * b[v, u] + a[j, i] * b[u, v]
    
    delta_self = new_self - old_self
    
    return int(delta_cross + delta_self)


def _two_opt_search(a, b, p, time_limit, start_time):
    """
    Perform 2-opt local search with first-improvement strategy.
    Uses a time-bounded loop with early termination.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break  # First improvement: restart scan from beginning
        
        if not improved:
            break
    
    return best_p, best_cost


def _greedy_init(a, b):
    """
    Greedy initialization: assign facilities to locations one by one
    choosing the assignment that minimizes incremental cost.
    """
    n = len(a)
    p = np.empty(n, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    
    # Precompute row sums for ordering
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Start with the facility having highest flow demand
    order = np.argsort(-a_row_sums)
    
    # For each facility in order, pick the location that minimizes incremental cost
    for idx in order:
        best_loc = -1
        best_cost = float('inf')
        
        for loc in range(n):
            if assigned[loc]:
                continue
            
            # Compute incremental cost of assigning facility idx to location loc
            inc_cost = 0
            for j in range(n):
                if assigned[j]:
                    # j is already assigned to location p[j]
                    inc_cost += a[idx, j] * b[loc, p[j]]
                    inc_cost += a[j, idx] * b[p[j], loc]
            
            if inc_cost < best_cost:
                best_cost = inc_cost
                best_loc = loc
        
        p[idx] = best_loc
        assigned[best_loc] = True
    
    return p


def _rank_matching_init(a, b):
    """
    Rank-matching heuristic: match high-flow facilities to low-distance locations.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def solve(a, b):
    """
    Solve the QAP instance tai256c.
    Returns a permutation p of 0..n-1 as a list of integers.
    """
    start_time = time.time()
    n = len(a)
    
    # Time budget: leave 15s margin for safety (150s total, use 135s)
    time_budget = 135
    
    best_p = None
    best_cost = float('inf')
    
    # Strategy 1: Rank-matching initialization + 2-opt
    p_rm = _rank_matching_init(a, b)
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 10:
        p_opt, cost_opt = _two_opt_search(a, b, p_rm, time_limit=remaining - 5, start_time=start_time)
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
    
    # Strategy 2: Greedy initialization + 2-opt
    p_greedy = _greedy_init(a, b)
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 10:
        p_opt, cost_opt = _two_opt_search(a, b, p_greedy, time_limit=remaining - 5, start_time=start_time)
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
    
    # Strategy 3: Random restarts with 2-opt
    rng = np.random.default_rng(12345)
    restart_count = 0
    while time.time() - start_time < time_budget - 20:
        restart_count += 1
        p_rand = rng.permutation(n)
        
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 15:
            break
        
        # Shorter time limit per restart
        per_restart_time = min(remaining - 5, 20)
        if per_restart_time < 5:
            break
        
        p_opt, cost_opt = _two_opt_search(a, b, p_rand, time_limit=per_restart_time, start_time=start_time)
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
    
    # Final safety check
    if time.time() - start_time > time_budget:
        # Shouldn't happen, but just in case
        pass
    
    if best_p is None:
        # Fallback: identity permutation
        best_p = np.arange(n)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
