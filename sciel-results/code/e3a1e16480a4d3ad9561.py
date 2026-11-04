import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in permutation p.
    O(n) time complexity.
    """
    pi = p[i]
    pj = p[j]
    
    b_pi_p = b[pi, p]
    b_pj_p = b[pj, p]
    b_p_pi = b[p, pi]
    b_p_pj = b[p, pj]
    
    # S1
    S1 = np.dot(a[i, :], b_pj_p) - np.dot(a[i, :], b_pi_p)
    C1i = a[i,i] * (b[pj, pi] - b[pi, pi])
    C1j = a[i,j] * (b[pj, pj] - b[pi, pj])
    
    # S2
    S2 = np.dot(a[:, i], b_p_pj) - np.dot(a[:, i], b_p_pi)
    C2i = a[i,i] * (b[pi, pj] - b[pi, pi])
    C2j = a[j,i] * (b[pj, pj] - b[pj, pi])
    
    # S3
    S3 = np.dot(a[j, :], b_pi_p) - np.dot(a[j, :], b_pj_p)
    C3i = a[j,i] * (b[pi, pi] - b[pj, pi])
    C3j = a[j,j] * (b[pi, pj] - b[pj, pj])
    
    # S4
    S4 = np.dot(a[:, j], b_p_pi) - np.dot(a[:, j], b_p_pj)
    C4i = a[i,j] * (b[pi, pi] - b[pi, pj])
    C4j = a[j,j] * (b[pj, pi] - b[pj, pj])
    
    delta_mixed = (S1 - C1i - C1j) + (S2 - C2i - C2j) + (S3 - C3i - C3j) + (S4 - C4i - C4j)
    
    delta_diag = a[i,i]*(b[pj,pj] - b[pi,pi]) + a[j,j]*(b[pi,pi] - b[pj,pj])
    delta_cross = a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    
    return int(delta_mixed + delta_diag + delta_cross)

def _first_improve_2opt(a, b, p, current_cost, time_limit=10.0):
    """
    First-improve 2-opt local search.
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                return p, cost
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    break
            if improved:
                break
                
    return p, cost

def _sa_run(a, b, p_init, current_cost, duration, T_start, T_end):
    """
    Run simulated annealing for a given duration.
    """
    n = len(a)
    start_time = time.time()
    p = p_init.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    while time.time() - start_time < duration:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        new_cost = cost + delta
        
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / duration)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Phase 1: Initial Solutions
    # Greedy
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_greedy = np.empty(n, dtype=np.int64)
    p_greedy[heavy] = central
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    # Random
    p_random = np.random.permutation(n).astype(np.int64)
    cost_random = _qap_cost(a, b, p_random)
    
    best_p = p_greedy.copy() if cost_greedy < cost_random else p_random.copy()
    best_cost = min(cost_greedy, cost_random)
    
    # Phase 2: Initial Polish with 2-Opt
    elapsed = time.time() - start_time
    polish_time = min(5.0, total_deadline - elapsed - 20.0)
    if polish_time > 0:
        best_p, best_cost = _first_improve_2opt(a, b, best_p, best_cost, time_limit=polish_time)
        
    # Phase 3: Multi-start SA
    elapsed = time.time() - start_time
    sa_time_total = total_deadline - elapsed - 5.0 # Reserve 5s for final LS
    num_sa_runs = 5
    sa_time_each = sa_time_total / num_sa_runs
    
    for k in range(num_sa_runs):
        if time.time() - start_time > total_deadline - 5.0:
            break
            
        # Diversify start
        if k == 0:
            p_start = best_p.copy()
            c_start = best_cost
        else:
            p_start = np.random.permutation(n).astype(np.int64)
            c_start = _qap_cost(a, b, p_start)
            # Quick polish
            p_start, c_start = _first_improve_2opt(a, b, p_start, c_start, time_limit=1.0)
        
        T_start = 1000.0
        T_end = 1.0
        p_sa, c_sa = _sa_run(a, b, p_start, c_start, sa_time_each, T_start, T_end)
        
        if c_sa < best_cost:
            best_cost = c_sa
            best_p = p_sa.copy()
            
    # Phase 4: Final Polish
    elapsed = time.time() - start_time
    final_ls_time = max(0.0, total_deadline - elapsed - 1.0)
    if final_ls_time > 0:
        best_p, best_cost = _first_improve_2opt(a, b, best_p, best_cost, time_limit=final_ls_time)
        
    return best_p.tolist()
